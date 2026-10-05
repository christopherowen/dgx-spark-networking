/* Compile with -Ithis-directory -pthread -fsanitize=address,undefined.
 * Includes the real C proxy. Only verbs and the GPU endpoint are simulated.
 * Payload source bytes are read at delivery, not at post time, to expose reuse.
 */
#include "../../sparknet/oneshot/_roce_proxy.c"
#include <assert.h>
#include <unistd.h>

struct transfer {
    struct transfer *next;
    uint8_t *source, *dest;
    size_t bytes;
    uint32_t inline_value;
    int inline_data, signaled;
    uint64_t id;
};
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static roce_ctx_t *ranks[4];
static uint32_t rng = 1;
static unsigned random_u32(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static int world, ring4, stripes, rotate;
static uint64_t messages;
static unsigned posted_in_round[4];
static uint32_t posted_seq[4];
static uint64_t direct_bytes[4][4], relay_bytes[4][4];
static unsigned written_seq[4][4][4096 / 16];
static unsigned char written_valid[4][4][4096 / 16];

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)bad;
    assert(qp && qp->source != qp->dest);
    if (ring4 == 1) assert((qp->source + 2) % 4 != qp->dest);
    if (rotate) {
        struct ibv_send_wr *last = wr;
        while (last->next) last = last->next;
        uint32_t seq;
        memcpy(&seq, (void *)(uintptr_t)last->sg_list->addr, sizeof(seq));
        int r = qp->source;
        if (!posted_in_round[r] || posted_seq[r] != seq) {
            assert(!posted_in_round[r] || posted_in_round[r] == 8);
            posted_seq[r] = seq; posted_in_round[r] = 0;
        }
        unsigned ordinal = posted_in_round[r]++;
        assert(ordinal < 8);
        assert(qp->hca == (r + (int)(seq & 3u) + (int)(ordinal % 4)) % 4);
        assert((qp->dest == (r+2)%4) == ((ordinal / 4 + (seq & 1u)) % 2));
    }
    pthread_mutex_lock(&lock);
    for (; wr; wr = wr->next) {
        assert(wr->num_sge == 1 && wr->opcode == IBV_WR_RDMA_WRITE);
        struct transfer *t = calloc(1, sizeof(*t));
        t->source = (uint8_t *)(uintptr_t)wr->sg_list->addr;
        t->dest = (uint8_t *)(uintptr_t)wr->wr.rdma.remote_addr;
        t->bytes = wr->sg_list->length;
        roce_ctx_t *c = ranks[qp->source], *d = ranks[qp->dest];
        assert(t->dest >= d->region && t->dest + t->bytes <= d->region + d->region_bytes);
        t->inline_data = !!(wr->send_flags & IBV_SEND_INLINE);
        t->signaled = !!(wr->send_flags & IBV_SEND_SIGNALED);
        if (t->inline_data) {
            assert(t->bytes == 4);
            memcpy(&t->inline_value, t->source, 4);
        } else {
            assert(t->source >= c->region && t->source + t->bytes <= c->region + c->region_bytes);
            assert(wr->sg_list->lkey == c->hca[qp->hca].mr->lkey);
            if (ring4 == 1) {
                size_t offset = (size_t)(t->dest - d->region - d->recv_off);
                int origin = (int)(offset / (2 * d->slot_bytes));
                assert(origin >= 0 && origin < world);
                uint32_t seq;
                assert(wr->next && (wr->next->send_flags & IBV_SEND_INLINE));
                memcpy(&seq, (void *)(uintptr_t)wr->next->sg_list->addr, 4);
                if (origin == qp->source) {
                    assert(qp->dest == (origin + 1) % 4 || qp->dest == (origin + 3) % 4);
                    direct_bytes[origin][qp->dest] += t->bytes;
                } else {
                    assert(qp->dest == (origin + 2) % 4);
                    relay_bytes[qp->source][qp->dest] += t->bytes;
                }
                // Every pack reaches each consumer exactly once, even when
                // the two routes deliver flags/data at very different times.
                size_t begin = (offset % d->slot_bytes) / 16;
                for (size_t i = begin; i < begin + t->bytes / 16; i++) {
                    assert(!written_valid[origin][qp->dest][i] || written_seq[origin][qp->dest][i] != seq);
                    written_valid[origin][qp->dest][i] = 1;
                    written_seq[origin][qp->dest][i] = seq;
                }
            }
        }
        int path = -1;
        for (int i = 0; i < peer_width(world, qp->source, qp->dest, stripes); i++)
            if (c->route_hca[qp->dest][i] == qp->hca) path = i;
        assert(path >= 0);
        assert(wr->wr.rdma.rkey == c->peer_rkey[path][qp->dest]);
        t->id = wr->wr_id;
        if (qp->tail) qp->tail->next = t; else qp->head = t;
        qp->tail = t;
        messages++;
    }
    pthread_mutex_unlock(&lock);
    return 0;
}
int ibv_poll_cq(struct ibv_cq *cq, int max, struct ibv_wc *wc) {
    pthread_mutex_lock(&lock);
    int n = 0;
    while (n < max && cq->read != cq->write) wc[n++] = cq->entries[cq->read++ % 1024];
    pthread_mutex_unlock(&lock);
    return n;
}
static void deliver(struct ibv_qp *q) {
    if (!q || !q->head) return;
    struct transfer *t = q->head;
    if (t->inline_data) __atomic_store_n((uint32_t *)t->dest, t->inline_value, __ATOMIC_RELEASE);
    else memcpy(t->dest, t->source, t->bytes);
    if (t->signaled) {
        assert(q->cq->write - q->cq->read < 1024);
        q->cq->entries[q->cq->write++ % 1024] = (struct ibv_wc){.wr_id=t->id, .status=IBV_WC_SUCCESS};
    }
    q->head = t->next;
    if (!q->head) q->tail = NULL;
    free(t);
}
static void progress(void) {
    pthread_mutex_lock(&lock);
    // Independently delay each direction and stripe; preserve RC QP FIFO only.
    for (int i = 0; i < 8; i++) {
        int r = random_u32() % world, h = random_u32() % ranks[r]->n_hca, p = random_u32() % world;
        deliver(ranks[r]->hca[h].qp[p]);
    }
    pthread_mutex_unlock(&lock);
}
static uint32_t *flag(roce_ctx_t *c, int src, uint32_t seq, int lane) {
    return (uint32_t *)(c->region + c->flag_off + (((size_t)src * 2 + (seq & 1)) * flag_lanes(c) + lane) * ROCE_FLAG_STRIDE);
}
static int receive_lanes(int rank, int peer) {
    if (ring4 == 1 && peer == (rank + 2) % world) return 2 * stripes;
    return peer_width(world, rank, peer, stripes);
}
static uint8_t pattern(int source, uint32_t seq, size_t i) { return (uint8_t)(source * 53 + seq * 7 + i * 11); }
static unsigned size_at(uint32_t seq) {
    const unsigned sizes[] = {16, 32, 48, 4096, 128, 64, 4000};
    return sizes[seq % 7];
}
static void setup(uint32_t seed) {
    memset(posted_in_round, 0, sizeof(posted_in_round));
    memset(direct_bytes, 0, sizeof(direct_bytes));
    memset(relay_bytes, 0, sizeof(relay_bytes));
    memset(written_valid, 0, sizeof(written_valid));
    uint64_t layout[7];
    assert(!roce_layout(world, 4096, layout));
    for (int r = 0; r < world; r++) {
        roce_ctx_t *c = ranks[r] = calloc(1, sizeof(*c));
        c->world = world; c->rank = r; c->ring4 = ring4; c->n_stripes = stripes; c->mesh_rotate = rotate; c->n_hca = stripes == 4 ? 4 : 2 * stripes;
        c->slot_bytes=4096; c->region_bytes=layout[4]; c->region=calloc(1, layout[4]);
        c->recv_off=layout[0]; c->flag_off=layout[1]; c->send_off=layout[2]; c->ctrl_off=layout[3];
        memset(c->route_hca, 255, sizeof(c->route_hca));
        for (int h = 0; h < c->n_hca; h++) {
            c->hca[h].cq=calloc(1, sizeof(struct ibv_cq));
            c->hca[h].mr=calloc(1, sizeof(struct ibv_mr));
            c->hca[h].mr->lkey = c->hca[h].mr->rkey = 100 + r*4+h;
        }
        int edge=0;
        for (int p=0; p<world; p++) {
            for (int lane=0; lane<flag_lanes(c); lane++)
                for (int slot=0; slot<2; slot++) *flag(c,p,slot,lane)=seed;
            for (int lane=0; lane<stripes; lane++) {
                if (p==r || (ring4 == 1 && p==(r+2)%4) ||
                    lane >= peer_width(world, r, p, stripes)) continue;
                int h = stripes == 4 ? (p == (r+2)%4 ? lane :
                    (p == (r+1)%4 ? 0 : 2) + lane) : (edge % 2)*stripes+lane;
                c->route_hca[p][lane]=h;
                struct ibv_qp *q=c->hca[h].qp[p]=calloc(1,sizeof(*q));
                q->source=r; q->dest=p; q->hca=h; q->cq=c->hca[h].cq; q->qp_num=1+r*16+h*4+p;
            }
            if (p!=r && (ring4 != 1 || p!=(r+2)%4)) edge++;
        }
        *(uint32_t *)(c->region+c->ctrl_off)=seed;
    }
    roce_blob_t blobs[4];
    for(int r=0;r<world;r++) assert(!roce_local_blob(ranks[r], &blobs[r], sizeof(blobs[r])));
    blobs[2].abi--;
    assert(roce_connect(ranks[0],blobs,sizeof(blobs))==-1);
    blobs[2].abi++;
    // Mixed geometry must be rejected, including the rank without a direct QP.
    blobs[2].ring4 ^= 1;
    assert(roce_connect(ranks[0],blobs,sizeof(blobs))==-1);
    blobs[2].ring4 ^= 1;
    blobs[2].mesh_rotate ^= 1;
    assert(roce_connect(ranks[0],blobs,sizeof(blobs))==-1);
    blobs[2].mesh_rotate ^= 1;
    for(int r=0;r<world;r++) assert(!roce_connect(ranks[r],blobs,sizeof(blobs)));
}
static void cleanup(void) {
    for(int r=0;r<world;r++) roce_stop(ranks[r]);
    for(int r=0;r<world;r++) {
        roce_ctx_t *c=ranks[r]; uint8_t *region=c->region;
        // A failed/stop test may intentionally leave pending network traffic.
        for(int h=0;h<c->n_hca;h++) for(int p=0;p<world;p++) {
            struct ibv_qp *q=c->hca[h].qp[p];
            if(q) while(q->head) {struct transfer *t=q->head; q->head=t->next; free(t);}
        }
        roce_destroy(c); free(region);
    }
}
static void run(unsigned count, uint32_t seed) {
    setup(seed);
    uint32_t next[4]; unsigned done[4]={0}; int active[4]={0};
    for(int r=0;r<world;r++) {next[r]=seed+1; assert(!roce_start(ranks[r]));}
    // Stop rank 0's proxy until its GPU has rung two pending doorbells. This
    // exercises the actual catch-up path, rather than assuming lockstep hosts.
    roce_stop(ranks[0]);
    int resumed=0;
    unsigned remaining=count*world;
    while(remaining) {
        progress();
        uint32_t *doorbell=(uint32_t *)(ranks[0]->region+ranks[0]->ctrl_off);
        if(!resumed && __atomic_load_n(doorbell,__ATOMIC_ACQUIRE)==seed+2) {
            assert(!roce_start(ranks[0])); resumed=1;
        }
        for(int r=0;r<world;r++) {
            roce_ctx_t *c=ranks[r]; assert(!roce_failed(c));
            uint32_t seq=next[r]; unsigned bytes=size_at(seq);
            if(!active[r] && done[r]<count && random_u32()%5==0) {
                uint8_t *send=c->region+c->send_off+(seq&1)*c->slot_bytes;
                for(unsigned i=0;i<bytes;i++) send[i]=pattern(r,seq,i);
                uint32_t *ctrl=(uint32_t *)(c->region+c->ctrl_off);
                __atomic_store_n(&ctrl[4+(seq&1)],bytes,__ATOMIC_RELEASE);
                __atomic_store_n(ctrl,seq,__ATOMIC_RELEASE); active[r]=1;
            }
            if(active[r]) {
                int ready=1;
                for(int p=0;p<world;p++) if(p!=r) for(int l=0;l<receive_lanes(r,p);l++)
                    ready &= __atomic_load_n(flag(c,p,seq,l),__ATOMIC_ACQUIRE)==seq;
                if(ready) {
                    for(int p=0;p<world;p++) if(p!=r) {
                        uint8_t *recv=c->region+c->recv_off+((size_t)p*2+(seq&1))*c->slot_bytes;
                        for(unsigned i=0;i<bytes;i++) assert(recv[i]==pattern(p,seq,i));
                    }
                    done[r]++; next[r]++; active[r]=0; remaining--;
                }
            }
        }
    }
    // Wait for the final operation to be posted on every proxy before teardown.
    // GPU completion alone need not mean its own host proxy has finished.
    for(int i=0;i<1000;i++) progress();
    if (ring4 == 1) {
        uint64_t total = 0;
        for (unsigned i=1;i<=count;i++) total += size_at(seed+i);
        for (int r=0;r<4;r++) {
            int next=(r+1)%4, prev=(r+3)%4;
            assert(direct_bytes[r][next] == total && direct_bytes[r][prev] == total);
            assert(relay_bytes[r][next] + relay_bytes[r][prev] == total);
            assert(relay_bytes[r][next] && relay_bytes[r][prev]);
            uint64_t a=relay_bytes[r][next], b=relay_bytes[r][prev];
            // At most one 16-byte remainder per lane and collective.
            assert((a>b?a-b:b-a) <= (uint64_t)count*stripes*16);
        }
    }
    cleanup();
}
static void failures(void) {
    setup(0); roce_ctx_t *c=ranks[0];
    atomic_store(&c->running,1);
    assert(post_op(c,1,4096+16)==-1);
    atomic_store(&c->running,0);
    assert(!roce_start(c));
    uint32_t *ctrl=(uint32_t *)(c->region+c->ctrl_off);
    __atomic_store_n(&ctrl[5],16,__ATOMIC_RELEASE);
    __atomic_store_n(&ctrl[0],1,__ATOMIC_RELEASE);
    usleep(10000); roce_stop(c); // predecessor never publishes: stop must unblock.
    assert(roce_failed(c));
    cleanup();
    setup(0); c=ranks[0];
    atomic_store(&c->running,1);
    ctrl=(uint32_t *)(c->region+c->ctrl_off); ctrl[2]=1;
    assert(relay_both_directions(c,1,16)==-1); atomic_store(&c->running,0); cleanup();
    setup(0); c=ranks[0]; atomic_store(&c->running,1);
    __atomic_store_n(flag(c,3,1,0),3,__ATOMIC_RELEASE);
    assert(relay_both_directions(c,1,16)==-1); atomic_store(&c->running,0); cleanup();
}

static void independent_directions(void) {
    // Publish only one neighbour's input. It must be forwarded while the
    // other neighbour is still absent, in BOTH choices of delayed direction.
    for (int ready_peer=1;ready_peer<=3;ready_peer+=2) {
        setup(0);
        roce_ctx_t *c=ranks[0];
        const uint32_t seq=1, bytes=128;
        uint8_t *recv=c->region+c->recv_off+((size_t)ready_peer*2+1)*c->slot_bytes;
        for (unsigned i=0;i<bytes;i++) recv[i]=pattern(ready_peer,seq,i);
        for (int lane=0;lane<stripes;lane++)
            __atomic_store_n(flag(c,ready_peer,seq,lane),seq,__ATOMIC_RELEASE);
        uint32_t *ctrl=(uint32_t *)(c->region+c->ctrl_off);
        assert(!roce_start(c));
        __atomic_store_n(&ctrl[5],bytes,__ATOMIC_RELEASE);
        __atomic_store_n(ctrl,seq,__ATOMIC_RELEASE);
        int dest=(ready_peer+2)%4, direction=ready_peer==3?0:1, ready=0;
        for (int attempt=0;attempt<10000 && !ready;attempt++) {
            progress();
            ready=1;
            for (int lane=0;lane<stripes;lane++)
                ready &= __atomic_load_n(flag(ranks[dest],ready_peer,seq,2*lane+direction),__ATOMIC_ACQUIRE)==seq;
            if (!ready) usleep(10);
        }
        assert(ready);
        for (int lane=0;lane<stripes;lane++)
            assert(__atomic_load_n(flag(ranks[dest],ready_peer,seq,2*lane+1-direction),__ATOMIC_ACQUIRE)==0);
        roce_stop(c); // Other direction still missing; stop must unblock it.
        assert(roce_failed(c));
        cleanup();
    }
}

static void placement(void) {
    // SPARKNET_ROCE_PROXY_CPU parsing, and (on Linux) that a pinned proxy thread
    // reports the CPU it was placed on.
    char err[160];
    assert(resolve_proxy_cpu(NULL, err, sizeof err) == -1);
    assert(resolve_proxy_cpu("", err, sizeof err) == -1);
    assert(resolve_proxy_cpu("none", err, sizeof err) == -1);
    assert(resolve_proxy_cpu("0", err, sizeof err) == 0);
    assert(resolve_proxy_cpu("x", err, sizeof err) == -2 && strstr(err, "SPARKNET_ROCE_PROXY_CPU"));
    assert(resolve_proxy_cpu("-1", err, sizeof err) == -2);
    assert(resolve_proxy_cpu("1000000", err, sizeof err) == -2);
    int big = resolve_proxy_cpu("big", err, sizeof err);
    assert(big >= 0 || strstr(err, "cpu_capacity"));  // hosts without cpu_capacity refuse "big"
#ifdef __linux__
    world = 3; ring4 = 0; stripes = 1; rotate = 0;
    setup(0);
    ranks[0]->proxy_cpu_plus1 = 1;
    assert(!roce_start(ranks[0]));
    for (int i = 0; i < 10000 && roce_stat(ranks[0], 6) == 0; i++) usleep(100);
    assert(roce_stat(ranks[0], 5) == 1 && roce_stat(ranks[0], 6) == 1);
    assert(!roce_failed(ranks[0]));
    cleanup();
#endif
}

int main(void) {
    for(stripes=1;stripes<=2;stripes++) {
        world=3;ring4=0;run(1000,0);
        world=4;ring4=1;run(3000,0);run(1000,UINT32_MAX-100);
        failures(); independent_directions();
        world=4;ring4=2;run(3000,0);run(1000,UINT32_MAX-100);
    }
    stripes=4;world=4;ring4=2;run(3000,0);run(1000,UINT32_MAX-100);
    rotate=1;run(3000,0);run(1000,UINT32_MAX-100);rotate=0;
    placement();
    printf("PASS: direct3/ring4/mesh4, 1/2/4 paths, 26000 collectives, bidirectional byte balance, independent directions, no duplicate packs, wrap, delayed DMA, stop/errors, thread placement (%llu writes)\n", (unsigned long long)messages);
}

/* Setup/teardown stubs. Contexts are supplied by setup; no fake successful NIC discovery. */
const char *ibv_wc_status_str(int s) {(void)s;return "fake completion error";}
struct ibv_device **ibv_get_device_list(int *n) {*n=0;return NULL;}
const char *ibv_get_device_name(struct ibv_device *p) {(void)p;return "none";}
void ibv_free_device_list(struct ibv_device **p) {(void)p;}
struct ibv_context *ibv_open_device(struct ibv_device *p) {(void)p;return NULL;}
int ibv_query_port(struct ibv_context *c,int p,struct ibv_port_attr *a) {(void)c;(void)p;(void)a;return -1;}
int ibv_query_gid(struct ibv_context *c,int p,int i,union ibv_gid *g) {(void)c;(void)p;(void)i;(void)g;return -1;}
struct ibv_pd *ibv_alloc_pd(struct ibv_context *c) {(void)c;return NULL;}
struct ibv_mr *ibv_reg_mr(struct ibv_pd *p,void *a,size_t n,int f) {(void)p;(void)a;(void)n;(void)f;return NULL;}
struct ibv_cq *ibv_create_cq(struct ibv_context *c,int n,void *a,void *b,int v) {(void)c;(void)n;(void)a;(void)b;(void)v;return NULL;}
struct ibv_qp *ibv_create_qp(struct ibv_pd *p,struct ibv_qp_init_attr *a) {(void)p;(void)a;return NULL;}
int ibv_modify_qp(struct ibv_qp *q,struct ibv_qp_attr *a,int f) {(void)f;assert(q); if(a->qp_state==IBV_QPS_RTR) {assert(a->dest_qp_num); assert(a->ah_attr.grh.flow_label == ((ring4 == 2 && q->dest == (q->source+2)%4) ? 16383 : 0));} return 0;}
int ibv_destroy_qp(struct ibv_qp *p) {free(p);return 0;}
int ibv_destroy_cq(struct ibv_cq *p) {free(p);return 0;}
int ibv_dereg_mr(struct ibv_mr *p) {free(p);return 0;}
int ibv_dealloc_pd(struct ibv_pd *p) {free(p);return 0;}
int ibv_close_device(struct ibv_context *p) {free(p);return 0;}
