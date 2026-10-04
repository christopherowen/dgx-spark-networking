/* Test-only verbs surface. The simulator executes the production proxy C. */
#pragma once
#include <stdint.h>
#include <stddef.h>
enum ibv_mtu { IBV_MTU_4096 = 5 };
enum { IBV_ACCESS_LOCAL_WRITE=1, IBV_ACCESS_REMOTE_WRITE=2,
       IBV_SEND_INLINE=1, IBV_SEND_SIGNALED=2, IBV_WC_SUCCESS=0,
       IBV_QPS_INIT=1, IBV_QPS_RTR=2, IBV_QPS_RTS=3, IBV_QPT_RC=1,
       IBV_PORT_ACTIVE=4, IBV_WR_RDMA_WRITE=1 };
#define IBV_QP_STATE 1
#define IBV_QP_AV 2
#define IBV_QP_PATH_MTU 4
#define IBV_QP_DEST_QPN 8
#define IBV_QP_RQ_PSN 16
#define IBV_QP_MAX_DEST_RD_ATOMIC 32
#define IBV_QP_MIN_RNR_TIMER 64
#define IBV_QP_TIMEOUT 128
#define IBV_QP_RETRY_CNT 256
#define IBV_QP_RNR_RETRY 512
#define IBV_QP_SQ_PSN 1024
#define IBV_QP_MAX_QP_RD_ATOMIC 2048
#define IBV_QP_PKEY_INDEX 4096
#define IBV_QP_PORT 8192
#define IBV_QP_ACCESS_FLAGS 16384
union ibv_gid { unsigned char raw[16]; };
struct ibv_context { int unused; };
struct ibv_device { int unused; };
struct ibv_pd { int unused; };
struct ibv_mr { uint32_t lkey, rkey; };
struct ibv_wc { uint64_t wr_id; int status; uint32_t vendor_err; };
struct ibv_cq { struct ibv_wc entries[1024]; unsigned read, write; };
struct transfer;
struct ibv_qp {
    uint32_t qp_num;
    int source, dest, hca;
    struct ibv_cq *cq;
    struct transfer *head, *tail;
};
struct ibv_port_attr { int state; uint16_t lid; enum ibv_mtu active_mtu; };
struct ibv_qp_init_attr {
    struct ibv_cq *send_cq, *recv_cq;
    int qp_type;
    struct { int max_send_wr, max_recv_wr, max_send_sge, max_recv_sge, max_inline_data; } cap;
};
struct ibv_qp_attr {
    int qp_state, pkey_index, port_num, qp_access_flags;
    enum ibv_mtu path_mtu;
    uint32_t dest_qp_num, rq_psn, sq_psn;
    int max_dest_rd_atomic, min_rnr_timer, timeout, retry_cnt, rnr_retry, max_rd_atomic;
    struct {
        int is_global, dlid, sl, src_path_bits, port_num;
        struct { union ibv_gid dgid; int sgid_index, hop_limit, traffic_class, flow_label; } grh;
    } ah_attr;
};
struct ibv_sge { uint64_t addr; uint32_t length, lkey; };
struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr *next;
    struct ibv_sge *sg_list;
    int num_sge, opcode, send_flags;
    union { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
};
int ibv_post_send(struct ibv_qp *, struct ibv_send_wr *, struct ibv_send_wr **);
int ibv_poll_cq(struct ibv_cq *, int, struct ibv_wc *);
const char *ibv_wc_status_str(int);
struct ibv_device **ibv_get_device_list(int *);
const char *ibv_get_device_name(struct ibv_device *);
void ibv_free_device_list(struct ibv_device **);
struct ibv_context *ibv_open_device(struct ibv_device *);
int ibv_query_port(struct ibv_context *, int, struct ibv_port_attr *);
int ibv_query_gid(struct ibv_context *, int, int, union ibv_gid *);
struct ibv_pd *ibv_alloc_pd(struct ibv_context *);
struct ibv_mr *ibv_reg_mr(struct ibv_pd *, void *, size_t, int);
struct ibv_cq *ibv_create_cq(struct ibv_context *, int, void *, void *, int);
struct ibv_qp *ibv_create_qp(struct ibv_pd *, struct ibv_qp_init_attr *);
int ibv_modify_qp(struct ibv_qp *, struct ibv_qp_attr *, int);
int ibv_destroy_qp(struct ibv_qp *);
int ibv_destroy_cq(struct ibv_cq *);
int ibv_dereg_mr(struct ibv_mr *);
int ibv_dealloc_pd(struct ibv_pd *);
int ibv_close_device(struct ibv_context *);
