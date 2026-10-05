// RDMA proxy for the one-shot collectives (sparknet.oneshot).
//
// Derived from local-inference-lab/b12x b12x/comm/roce/_roce_proxy.c at
// f8069b2c plus the switchless, ring4 relay, mesh4 and dispatch patches
// (see docs/provenance.md). Wire ABI 10 is unchanged.
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[src][slot][lane] sequence number written after that route lane's
//                          payload stripe
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on flag[peer][seq & 1][lane] for every peer and route
// lane. The proxy thread below spins on ctrl.seq and, for every peer, stripes
// the payload across the HCAs selected for that peer. Each stripe is followed
// by its own 4-byte seq write on the same reliable QP, so its flag cannot become
// visible before its payload. The GPU waits for every stripe flag before
// consuming the receive slot. In explicit ring4 mode the host additionally
// forwards disjoint halves of each received stripe in both directions. The
// opposite GPU waits on twice as many flags; source slots and rank-order sums
// are unchanged. Neighbour traffic always uses its direct QPs.
//
// This file is compiled by sparknet.oneshot._proxy at first use with the host
// gcc and libibverbs; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef CPU_SETSIZE
#define CPU_SETSIZE 1024
#endif
#define ROCE_CPU_WORDS (CPU_SETSIZE / 64)

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_LOCAL_HCAS 4
#define ROCE_MAX_STRIPES 4
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
#define ROCE_ABI_VERSION 10
// Model graphs leave sub-millisecond gaps between collectives.  Keep the
// proxy hot across those gaps; sleeping there adds one scheduler wakeup to
// every collective on the graph's critical path.
#define ROCE_IDLE_SPINS 20000000

typedef struct {
    uint32_t abi, world, rank, ring4, mesh_rotate;
    uint64_t slot_bytes;
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_LOCAL_HCAS];
    uint16_t lid[ROCE_MAX_LOCAL_HCAS];
    uint8_t gid[ROCE_MAX_LOCAL_HCAS][16];
    uint32_t mtu[ROCE_MAX_LOCAL_HCAS];
    uint32_t qp_num[ROCE_MAX_LOCAL_HCAS][ROCE_MAX_PEERS];
    uint8_t n_hca;
    uint8_t n_stripes;
    uint8_t route_hca[ROCE_MAX_PEERS][ROCE_MAX_STRIPES];
} roce_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    uint32_t outstanding[ROCE_MAX_PEERS];
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
    int world;
    int rank;
    int ring4;
    int mesh_rotate;
    int n_hca;
    int n_stripes;
    int gid_index;
    int traffic_class;
    roce_hca_t hca[ROCE_MAX_LOCAL_HCAS];
    uint8_t route_hca[ROCE_MAX_PEERS][ROCE_MAX_STRIPES];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    int started;
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint32_t peer_rkey[ROCE_MAX_STRIPES][ROCE_MAX_PEERS];
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint64_t ops_posted;
    uint64_t writes_completed;
    // Proxy thread placement (SPARKNET_ROCE_PROXY_CPU): the CPUs the thread may
    // run on (a zeroed context is unpinned) and, plus one, the CPU it first
    // ran on.
    uint64_t proxy_cpu_mask[ROCE_CPU_WORDS];
    int proxy_cpu_count;
    int proxy_cpu_observed_plus1;
    char err[512];
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

static void mask_set(uint64_t *mask, long cpu) { mask[cpu / 64] |= (uint64_t)1 << (cpu % 64); }

static int mask_test(const uint64_t *mask, long cpu) { return (int)((mask[cpu / 64] >> (cpu % 64)) & 1u); }

static long cpu_capacity(long cpu) {
    char path[96];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpu_capacity", cpu);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    long capacity = -1;
    if (fscanf(f, "%ld", &capacity) != 1) {
        capacity = -1;
    }
    fclose(f);
    return capacity;
}

// SPARKNET_ROCE_PROXY_CPU: unset, empty or "none" leaves the proxy thread to
// the scheduler; a CPU number pins it to that core; "big" confines it to the
// big-core cluster: every core whose sysfs cpu_capacity lies above the
// midpoint between the smallest and the largest capacity (the GB10 reports
// its ten Cortex-X925 cores at 997 to 1024 and its ten Cortex-A725 cores at
// 718 to 731; a homogeneous host selects every core). An unpinned poller can
// land on a little core, and a single-core pin would compete with whatever
// the serving process keeps on that core. Fills the mask and returns the
// number of CPUs selected (0 = unpinned), or -1 with err set.
static int resolve_proxy_cpus(const char *value, uint64_t *mask, char *err, size_t err_len) {
    memset(mask, 0, ROCE_CPU_WORDS * sizeof(uint64_t));
    if (value == NULL || value[0] == '\0' || strcmp(value, "none") == 0) {
        return 0;
    }
    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    if (ncpu > CPU_SETSIZE) {
        ncpu = CPU_SETSIZE;
    }
    if (strcmp(value, "big") == 0) {
        long best = -1, worst = -1;
        for (long cpu = 0; cpu < ncpu; cpu++) {
            long capacity = cpu_capacity(cpu);
            if (capacity < 0) {
                continue;
            }
            if (capacity > best) {
                best = capacity;
            }
            if (worst < 0 || capacity < worst) {
                worst = capacity;
            }
        }
        if (best < 0) {
            snprintf(err, err_len, "SPARKNET_ROCE_PROXY_CPU=big: no cpu_capacity under /sys/devices/system/cpu");
            return -1;
        }
        long threshold = (best + worst) / 2;
        int count = 0;
        for (long cpu = 0; cpu < ncpu; cpu++) {
            long capacity = cpu_capacity(cpu);
            if (capacity >= 0 && (best == worst || capacity > threshold)) {
                mask_set(mask, cpu);
                count++;
            }
        }
        return count;
    }
    char *end = NULL;
    long cpu = strtol(value, &end, 10);
    if (end == value || *end != '\0' || cpu < 0 || cpu >= ncpu) {
        snprintf(err, err_len, "SPARKNET_ROCE_PROXY_CPU must be none, big or a CPU number below %ld", ncpu);
        return -1;
    }
    mask_set(mask, cpu);
    return 1;
}

// Name the proxy thread and apply the requested placement from inside it.
static int place_proxy_thread(roce_ctx_t *c) {
#ifdef __linux__
    pthread_setname_np(pthread_self(), "sparknet-proxy");
    if (c->proxy_cpu_count > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (long cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (mask_test(c->proxy_cpu_mask, cpu)) {
                CPU_SET(cpu, &set);
            }
        }
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (rc != 0) {
            set_err(c, "pthread_setaffinity_np", rc);
            return -1;
        }
    }
    int cpu = sched_getcpu();
    c->proxy_cpu_observed_plus1 = cpu >= 0 ? cpu + 1 : 0;
#else
    (void)c;
#endif
    return 0;
}

int roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots}
    if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0) {
        return -1;
    }
    // Reject a layout whose arithmetic would wrap; the caller sizes slots from
    // configuration, so a wrapped region must fail here rather than at the NIC.
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, total;
    if (slot_bytes > ((uint64_t)1 << 40) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow(
            (uint64_t)world * ROCE_SLOTS * ROCE_MAX_STRIPES,
            (uint64_t)ROCE_FLAG_STRIDE,
            &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    uint64_t recv_off = 0;
    uint64_t flag_off = recv_off + recv_bytes;
    out[0] = recv_off;
    out[1] = flag_off;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

// Modes: 0 direct, 1 CPU relay, 2 NIC-forwarded mesh. Only mode 1 omits opposite QPs.
static int adjacent(int world, int rank, int peer, int ring4) {
    return peer != rank && (ring4 != 1 || peer == (rank + 1) % world ||
                            peer == (rank + world - 1) % world);
}

static int open_hca(roce_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index, &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        if (!adjacent(c->world, c->rank, p, c->ring4)) {
            continue;
        }
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = hca->cq;
        attr.recv_cq = hca->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        hca->qp[p] = ibv_create_qp(hca->pd, &attr);
        if (hca->qp[p] == NULL) {
            set_err(c, "ibv_create_qp", errno);
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = ROCE_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(hca->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_err(c, "ibv_modify_qp(INIT)", rc);
            return -1;
        }
    }
    return 0;
}

static int peer_width(int world, int rank, int peer, int stripes) {
    return stripes == 4 && peer != (rank + 2) % world ? 2 : stripes;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names, int n_hca,
                        const int *route_hca, int n_stripes, int gid_index,
                        int traffic_class, int ring4,
                        void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, char *err, uint64_t err_len) {
    uint64_t layout[7];
    if (roce_layout(world, slot_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_LOCAL_HCAS ||
        n_stripes < 1 || n_stripes > ROCE_MAX_STRIPES || route_hca == NULL ||
        traffic_class < 0 || traffic_class > 255 ||
        (ring4 < 0 || ring4 > 2) || (ring4 && world != 4) ||
        (n_stripes > 2 && (n_stripes != 4 || ring4 != 2 || n_hca != 4))) {
        snprintf(err, err_len, "invalid roce runtime geometry");
        return NULL;
    }
    for (int p = 0; p < world; p++) {
        if (!adjacent(world, rank, p, ring4)) {
            for (int lane = 0; lane < n_stripes; lane++) {
                if (route_hca[p * n_stripes + lane] != -1) {
                    snprintf(err, err_len, "non-neighbour rank %d has a route", p);
                    return NULL;
                }
            }
            continue;
        }
        for (int lane = 0; lane < n_stripes; lane++) {
            int h = route_hca[p * n_stripes + lane];
            if (lane >= peer_width(world, rank, p, n_stripes)) {
                if (h != -1) {
                    snprintf(err, err_len, "unused neighbor path has a route");
                    return NULL;
                }
                continue;
            }
            if (h < 0 || h >= n_hca) {
                snprintf(err, err_len,
                         "invalid HCA route for peer %d lane %d: %d", p, lane, h);
                return NULL;
            }
            for (int prior = 0; prior < lane; prior++) {
                if (h == route_hca[p * n_stripes + prior]) {
                    snprintf(err, err_len,
                             "duplicate HCA route for peer %d: %d", p, h);
                    return NULL;
                }
            }
        }
    }
    if (n_stripes == 4) {
        unsigned used = 0;
        for (int p = 0; p < world; p++) {
            if (p == rank || p == (rank + 2) % world) continue;
            for (int lane = 0; lane < 2; lane++) {
                unsigned bit = 1u << route_hca[p * n_stripes + lane];
                if (used & bit) {
                    snprintf(err, err_len, "neighbor paths must cover four distinct HCAs");
                    return NULL;
                }
                used |= bit;
            }
        }
    }
    roce_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    const char *rotate = getenv("SPARKNET_ROCE_MESH_ROTATE");
    if (rotate && strcmp(rotate, "0") && strcmp(rotate, "1")) {
        snprintf(err, err_len, "SPARKNET_ROCE_MESH_ROTATE must be 0 or 1");
        free(c); return NULL;
    }
    c->mesh_rotate = rotate && !strcmp(rotate, "1");
    if (c->mesh_rotate && n_stripes != 4) {
        snprintf(err, err_len, "rotating posts require four mesh paths");
        free(c); return NULL;
    }
    c->proxy_cpu_count = resolve_proxy_cpus(getenv("SPARKNET_ROCE_PROXY_CPU"), c->proxy_cpu_mask, err, err_len);
    if (c->proxy_cpu_count < 0) {
        free(c); return NULL;
    }
#ifndef __linux__
    if (c->proxy_cpu_count > 0) {
        snprintf(err, err_len, "SPARKNET_ROCE_PROXY_CPU: thread pinning is only supported on Linux");
        free(c); return NULL;
    }
#endif
    c->world = world;
    c->rank = rank;
    c->ring4 = ring4;
    c->n_hca = n_hca;
    c->n_stripes = n_stripes;
    c->gid_index = gid_index;
    c->traffic_class = traffic_class;
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    for (int p = 0; p < world; p++) {
        for (int lane = 0; lane < n_stripes; lane++) {
            c->route_hca[p][lane] = (uint8_t)route_hca[p * n_stripes + lane];
        }
    }
    for (int h = 0; h < n_hca; h++) {
        if (open_hca(c, h, hca_names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(roce_blob_t)) {
        return -1;
    }
    roce_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.abi = ROCE_ABI_VERSION;
    blob.world = c->world;
    blob.rank = c->rank;
    blob.ring4 = c->ring4;
    blob.mesh_rotate = c->mesh_rotate;
    blob.slot_bytes = c->slot_bytes;
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    blob.n_hca = (uint8_t)c->n_hca;
    blob.n_stripes = (uint8_t)c->n_stripes;
    memcpy(blob.route_hca, c->route_hca, sizeof(blob.route_hca));
    for (int h = 0; h < c->n_hca; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            blob.qp_num[h][p] = c->hca[h].qp[p] ? c->hca[h].qp[p]->qp_num : 0;
        }
    }
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

static int connect_qp(roce_ctx_t *c, int local_h, int remote_h, int p,
                      const roce_blob_t *peer) {
    roce_hca_t *hca = &c->hca[local_h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[remote_h] < (uint32_t)hca->mtu
                                      ? peer->mtu[remote_h]
                                      : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[remote_h][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[remote_h];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[remote_h], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
    // mlx5 maps flow label 16383 to UDP source 65535. The host fabric
    // marks only this reserved source port for intermediate NIC forwarding.
    rtr.ah_attr.grh.flow_label =
        (c->ring4 == 2 && p == (c->rank + 2) % 4) ? 16383 : 0;
    int rc = ibv_modify_qp(hca->qp[p], &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(hca->qp[p], &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const roce_blob_t *all = (const roce_blob_t *)blobs;
    // Validate every record before connecting anything, including the opposite
    // rank whose QP is deliberately absent. Mixed modes cannot silently hang.
    for (int p = 0; p < c->world; p++) {
        if (all[p].abi != ROCE_ABI_VERSION || all[p].world != (uint32_t)c->world ||
            all[p].rank != (uint32_t)p || all[p].ring4 != (uint32_t)c->ring4 ||
            all[p].slot_bytes != c->slot_bytes ||
            all[p].n_hca < 1 || all[p].n_hca > ROCE_MAX_LOCAL_HCAS ||
            all[p].n_stripes != c->n_stripes ||
            all[p].mesh_rotate != (uint32_t)c->mesh_rotate) {
            snprintf(c->err, sizeof(c->err), "rank %d published incompatible transport geometry", p);
            return -1;
        }
        for (int q = 0; q < c->world; q++) {
            for (int lane = 0; lane < c->n_stripes; lane++) {
                unsigned h = all[p].route_hca[q][lane];
                if ((adjacent(c->world, p, q, c->ring4) &&
                     lane < peer_width(c->world, p, q, c->n_stripes))
                        ? (h >= all[p].n_hca || !all[p].qp_num[h][q])
                        : h != UINT8_MAX) {
                    snprintf(c->err, sizeof(c->err), "rank %d published invalid route to %d", p, q);
                    return -1;
                }
            }
        }
    }
    for (int p = 0; p < c->world; p++) {
        if (!adjacent(c->world, c->rank, p, c->ring4)) continue;
        c->peer_addr[p] = all[p].region_addr;
        for (int lane = 0; lane < peer_width(c->world, c->rank, p, c->n_stripes); lane++) {
            int local_h = c->route_hca[p][lane];
            int remote_h = all[p].route_hca[c->rank][lane];
            if (remote_h < 0 || remote_h >= all[p].n_hca) {
                snprintf(c->err, sizeof(c->err),
                         "rank %d published invalid HCA route for lane %d", p, lane);
                return -1;
            }
            c->peer_rkey[lane][p] = all[p].rkey[remote_h];
            if (connect_qp(c, local_h, remote_h, p, &all[p]) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int drain_cq(roce_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
                     (unsigned)wc[i].wr_id, ibv_wc_status_str(wc[i].status),
                     wc[i].vendor_err, c->last_seq);
            return -1;
        }
        c->hca[h].outstanding[wc[i].wr_id] -= 1;
        c->hca[h].writes_completed += 1;
        c->writes_completed += 1;
    }
    return 0;
}

// Ring relays have two independently ordered fragments per physical lane.
// Neighbours publish one flag per lane; the opposite rank waits for both halves.
static int flag_lanes(const roce_ctx_t *c) {
    return c->n_stripes * (c->ring4 == 1 ? 2 : 1);
}

static int post_fragment(roce_ctx_t *c, uint32_t seq,
                         int origin, int p, uint8_t *send, int lane,
                         uint64_t byte_offset, uint32_t stripe_bytes, int flag_lane) {
    uint32_t slot = seq & 1u;
    uint32_t seq_copy = seq;
    {
        int h = c->route_hca[p][lane];
        roce_hca_t *hca = &c->hca[h];
        // Two work requests per stripe; keep the queue at most a quarter
        // full so a provider that needs extra entries can never fail a post.
        while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {
            if (drain_cq(c, h) != 0) {
                return -1;
            }
            // A peer that stopped acknowledging keeps the QP retrying for
            // a long time; honour a stop request instead of blocking
            // roce_stop (and so teardown) behind it.
            if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
                snprintf(c->err, sizeof(c->err),
                         "RoCE proxy stopped with %u writes outstanding to rank %d",
                         hca->outstanding[p], p);
                return -1;
            }
        }
        uint64_t remote = c->peer_addr[p];
        struct ibv_sge flag_sge = {
            .addr = (uint64_t)(uintptr_t)&seq_copy,
            .length = 4,
            .lkey = 0,
        };
        struct ibv_send_wr flag_wr;
        memset(&flag_wr, 0, sizeof(flag_wr));
        flag_wr.wr_id = (uint64_t)p;
        flag_wr.sg_list = &flag_sge;
        flag_wr.num_sge = 1;
        flag_wr.opcode = IBV_WR_RDMA_WRITE;
        flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        flag_wr.wr.rdma.remote_addr =
            remote + c->flag_off +
            (((uint64_t)origin * ROCE_SLOTS + slot) * (uint64_t)flag_lanes(c) +
             (uint64_t)flag_lane) * ROCE_FLAG_STRIDE;
        flag_wr.wr.rdma.rkey = c->peer_rkey[lane][p];

        struct ibv_send_wr data_wr;
        struct ibv_sge data_sge;
        struct ibv_send_wr *first_wr = &flag_wr;
        if (stripe_bytes != 0) {
            data_sge = (struct ibv_sge){
                .addr = (uint64_t)(uintptr_t)(send + byte_offset),
                .length = stripe_bytes,
                .lkey = hca->mr->lkey,
            };
            memset(&data_wr, 0, sizeof(data_wr));
            data_wr.wr_id = (uint64_t)p;
            data_wr.next = &flag_wr;
            data_wr.sg_list = &data_sge;
            data_wr.num_sge = 1;
            data_wr.opcode = IBV_WR_RDMA_WRITE;
            data_wr.wr.rdma.remote_addr =
                remote + c->recv_off +
                ((uint64_t)origin * ROCE_SLOTS + slot) * c->slot_bytes +
                byte_offset;
            data_wr.wr.rdma.rkey = c->peer_rkey[lane][p];
            first_wr = &data_wr;
        }
        struct ibv_send_wr *bad = NULL;
        int rc = ibv_post_send(hca->qp[p], first_wr, &bad);
        if (rc != 0) {
            set_err(c, "ibv_post_send", rc);
            return -1;
        }
        hca->outstanding[p] += 1;
        hca->bytes_posted += stripe_bytes;
    }
    return 0;
}

static void stripe_range(uint32_t nbytes, int width, int lane,
                         uint32_t *offset, uint32_t *packs) {
    uint32_t total = nbytes / 16, remainder = total % (uint32_t)width;
    *offset = (uint32_t)lane * (total / (uint32_t)width) +
        ((uint32_t)lane < remainder ? (uint32_t)lane : remainder);
    *packs = total / (uint32_t)width + ((uint32_t)lane < remainder);
}

static int post_path(roce_ctx_t *c, uint32_t seq, uint32_t nbytes,
                     int origin, int p, uint8_t *send, int lane) {
    uint32_t offset, packs;
    stripe_range(nbytes, peer_width(c->world, c->rank, p, c->n_stripes),
                 lane, &offset, &packs);
    return post_fragment(c, seq, origin, p, send, lane,
                         (uint64_t)offset * 16, packs * 16, lane);
}

static int post_peer(roce_ctx_t *c, uint32_t seq, uint32_t nbytes,
                     int origin, int p, uint8_t *send) {
    for (int lane = 0; lane < peer_width(c->world, c->rank, p, c->n_stripes); lane++)
        if (post_path(c, seq, nbytes, origin, p, send, lane) != 0) return -1;
    return 0;
}

static int relay_both_directions(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    unsigned pending = (1u << (2 * c->n_stripes)) - 1u;
    for (unsigned spins = 0;; spins++) {
        if (!atomic_load_explicit(&c->running, memory_order_relaxed) ||
            __atomic_load_n(&ctrl[2], __ATOMIC_ACQUIRE)) {
            snprintf(c->err, sizeof(c->err), "relay stopped or GPU failed at seq %u", seq);
            return -1;
        }
        // Do not wait for one direction before progressing the other. Each
        // fragment can forward as soon as its own first-hop lane has arrived.
        for (int fragment = 0; fragment < 2 * c->n_stripes; fragment++) {
            unsigned bit = 1u << fragment;
            if (!(pending & bit)) continue;
            int direction = fragment % 2, lane = fragment / 2;
            int prev = (c->rank + c->world - 1) % c->world;
            int next = (c->rank + 1) % c->world;
            int origin = direction == 0 ? prev : next;
            int destination = direction == 0 ? next : prev;
            uint32_t *flag = (uint32_t *)(c->region + c->flag_off +
                (((size_t)origin * ROCE_SLOTS + (seq & 1u)) * flag_lanes(c) + lane) * ROCE_FLAG_STRIDE);
            uint32_t observed = __atomic_load_n(flag, __ATOMIC_ACQUIRE);
            if ((int32_t)(observed - seq) > 0) {
                snprintf(c->err, sizeof(c->err), "relay slot overwritten: rank %d seq %u saw %u", origin, seq, observed);
                return -1;
            }
            if (observed != seq) continue;
            // Match rdma-core's device-to-CPU barrier after observing ordered
            // DMA flags; normal C acquire alone need not include the NIC domain.
#if defined(__aarch64__)
            __asm__ volatile("dmb oshld" ::: "memory");
#elif defined(__x86_64__)
            __asm__ volatile("lfence" ::: "memory");
#else
            __sync_synchronize();
#endif
            uint32_t offset, packs;
            stripe_range(nbytes, c->n_stripes, lane, &offset, &packs);
            // Split each physical stripe into disjoint 16-byte packs. Rotate
            // the odd pack across epochs and lanes instead of biasing one path.
            uint32_t first = (packs + ((seq + (uint32_t)lane) & 1u)) / 2;
            if (direction == 0) packs = first;
            else { offset += first; packs -= first; }
            uint8_t *received = c->region + c->recv_off +
                ((size_t)origin * ROCE_SLOTS + (seq & 1u)) * c->slot_bytes;
            // Preserve the original source's slot. Its N+2 cannot begin until
            // the opposite GPU has consumed BOTH fragments of N. Zero-length
            // fragments still publish a flag, including the 16-byte case.
            if (post_fragment(c, seq, origin, destination, received, lane,
                              (uint64_t)offset * 16, packs * 16, fragment) != 0)
                return -1;
            pending &= ~bit;
        }
        if (!pending) return 0;
        if ((spins & 63u) == 0) {
            for (int h = 0; h < c->n_hca; h++)
                if (drain_cq(c, h) != 0) return -1;
        }
        if ((spins & 4095u) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            if ((now.tv_sec - start.tv_sec) * 1000000000LL + now.tv_nsec - start.tv_nsec >= 5000000000LL) {
                snprintf(c->err, sizeof(c->err), "relay timed out: seq %u pending 0x%x", seq, pending);
                return -1;
            }
        }
    }
}

static int post_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (nbytes == 0 || nbytes % 16 != 0 || nbytes > c->slot_bytes) {
        snprintf(c->err, sizeof(c->err), "invalid RoCE payload size %u (slot %zu)", nbytes, c->slot_bytes);
        return -1;
    }
    uint32_t slot = seq & 1u;
    uint8_t *send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
    if (c->mesh_rotate) {
        // Equal byte striping is independent of submission order. Give each
        // HCA one write before the next round; rotate first HCA and class.
        int opposite = (c->rank + 2) % c->world;
        int first = (c->rank + (int)(seq & 3u)) % c->n_hca;
        for (int round = 0; round < 2; round++) {
            int want_opposite = (round + (int)(seq & 1u)) % 2;
            for (int offset = 0; offset < c->n_hca; offset++) {
                int h = (first + offset) % c->n_hca;
                for (int p = 0; p < c->world; p++) {
                    if (p == c->rank || ((p == opposite) != want_opposite)) continue;
                    for (int lane = 0; lane < peer_width(c->world, c->rank, p, c->n_stripes); lane++)
                        if (c->route_hca[p][lane] == h &&
                            post_path(c, seq, nbytes, c->rank, p, send, lane) != 0) return -1;
                }
            }
        }
    } else {
    for (int p = 0; p < c->world; p++) {
        if (adjacent(c->world, c->rank, p, c->ring4) &&
            post_peer(c, seq, nbytes, c->rank, p, send) != 0) return -1;
    }
    }
    if (c->ring4 == 1) {
        if (relay_both_directions(c, seq, nbytes) != 0) return -1;
    }
    c->ops_posted += 1;
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

static void *proxy_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    if (place_proxy_thread(c) != 0) {
        atomic_store(&c->failed, 1);
        return NULL;
    }
    // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
    // doorbell, request a short nanosleep between polls (the OS decides the
    // actual delay) so an idle runtime does not hold a core next to the
    // serving process.  The missed-doorbell catch-up below keeps the protocol
    // correct however long the thread is away.
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
        if (seq == c->last_seq) {
            idle++;
            if (idle % 64 == 0) {
                for (int h = 0; h < c->n_hca; h++) {
                    if (drain_cq(c, h) != 0) {
                        atomic_store(&c->failed, 1);
                        return NULL;
                    }
                }
            }
            if (idle >= ROCE_IDLE_SPINS) {
                nanosleep(&nap, NULL);
            }
            continue;
        }
        idle = 0;
        // The doorbell holds only the newest sequence.  Our kernel for op N
        // completes on the peers' payloads alone, so op N+1 can ring before
        // this thread has seen op N (it slept, or the scheduler moved it).
        // Peers cannot get further than one op ahead of us, so at most
        // ROCE_SLOTS doorbells are pending and every send slot is intact:
        // post each missed sequence in order using its per-slot byte count.
        uint32_t pending = seq - c->last_seq;
        if (pending > ROCE_SLOTS) {
            snprintf(c->err, sizeof(c->err),
                     "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            atomic_store(&c->failed, 1);
            return NULL;
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t nbytes = ctrl[4 + (s & 1u)];
            if (post_op(c, s, nbytes) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->last_seq = s;
        }
    }
    return NULL;
}

int roce_start(roce_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        // A restart continues from the last posted sequence so ops that rang
        // the doorbell while the thread was stopped are still posted.
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[0];
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void roce_stop(roce_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
        pthread_join(c->thread, NULL);
    }
}

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    case 3:
        return c->n_stripes;
    case 4:
        return c->mesh_rotate;
    case 5:
        return (uint64_t)c->proxy_cpu_count;
    case 6:
        return (uint64_t)c->proxy_cpu_observed_plus1;
    default:
        return 0;
    }
}

// The CPUs the proxy thread is pinned to, in ascending order; returns how many were written.
int roce_proxy_cpus(roce_ctx_t *c, int *out, int max) {
    int n = 0;
    for (long cpu = 0; cpu < CPU_SETSIZE && n < max; cpu++) {
        if (mask_test(c->proxy_cpu_mask, cpu)) {
            out[n++] = (int)cpu;
        }
    }
    return n;
}

uint64_t roce_hca_stat(roce_ctx_t *c, int hca, int which) {
    if (hca < 0 || hca >= c->n_hca) {
        return -1;
    }
    switch (which) {
    case 0:
        return c->hca[hca].writes_completed;
    case 1:
        return c->hca[hca].bytes_posted;
    default:
        return 0;
    }
}

void roce_destroy(roce_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    roce_stop(c);
    for (int h = 0; h < ROCE_MAX_LOCAL_HCAS; h++) {
        roce_hca_t *hca = &c->hca[h];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
            if (hca->qp[p] != NULL) {
                ibv_destroy_qp(hca->qp[p]);
            }
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
    free(c);
}
