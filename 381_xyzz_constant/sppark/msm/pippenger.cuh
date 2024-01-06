// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>


#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef NTHREADS
# define NTHREADS 64
#endif
#if NTHREADS < 32 || (NTHREADS & (NTHREADS-1)) != 0
# error "bad NTHREADS value"
#endif

constexpr static int log2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static const int NTHRBITS = log2(NTHREADS);

#ifndef NBITS
# define NBITS 253
#endif
#ifndef WBITS
# define WBITS 16
#endif
#define NWINS 16  // ((NBITS+WBITS-1)/WBITS)   // ceil(NBITS/WBITS)

#ifndef LARGE_L1_CODE_CACHE
# define LARGE_L1_CODE_CACHE 0
#endif

__global__
void pre_compute(affine_t* pre_points, size_t npoints);

__global__
void process_scalar_1(uint16_t* scalar, uint32_t* scalar_tuple,
                      uint32_t* d_scalar_map, uint32_t* point_idx, size_t npoints);

__global__
void process_scalar_2(uint32_t* scalar_tuple_out,
                      uint16_t* bucket_idx, size_t npoints);

__global__
void buffer_init(bucket_t *buckets_pre, uint16_t* bucket_idx_pre_vector, bucket_t *buckets);

__global__
void bucket_acc(uint32_t* scalar_tuple_out, uint16_t* bucket_idx, uint32_t* point_idx_out,
                affine_t* pre_points, bucket_t *buckets_pre,
                uint16_t* bucket_idx_pre_vector, size_t npoints);

__global__
void bucket_acc_2(bucket_t *buckets_pre, uint16_t* bucket_idx_pre_vector, bucket_t *buckets, uint32_t* d_op_index);

__global__
void bucket_agg_1(bucket_t *buckets);

__global__
void bucket_agg_2(bucket_t *buckets);

__global__
void recursive_sum(bucket_t *buckets, bucket_t *res);


#ifdef __CUDA_ARCH__

#include <cooperative_groups.h>

static __shared__ bucket_t bucket_acc_smem[NTHREADS * 2];

// Transposed scalar_t
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ uint32_t& operator[](size_t i)              { return val[i][0]; }
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

class scalars_T {
    scalar_T* ptr;

public:
    __device__ scalars_T(void* rhs) { ptr = (scalar_T*)rhs; }
    __device__ scalar_T& operator[](size_t i)
    {   return *(scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
    __device__ const scalar_T& operator[](size_t i) const
    {   return *(const scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
};

constexpr static __device__ int dlog2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }


#if WBITS==16
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t ret = d[off/32];
    return (ret >> (off%32)) & ((1<<bits) - 1);
}
#else
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];

    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
}
#endif


static __device__ uint32_t max_bits(uint32_t scalar)
{
    uint32_t max = 32;
    return max;
}

static __device__ bool test_bit(uint32_t scalar, uint32_t bitno)
{
    if (bitno >= 32)
        return false;
    return ((scalar >> bitno) & 0x1);
}

template<class bucket_t>
static __device__ void mul(bucket_t& res, const bucket_t& base, uint32_t scalar)
{
    res.inf();

    bool found_one = false;
    uint32_t mb = max_bits(scalar);
    for (int32_t i = mb - 1; i >= 0; --i)
    {
        if (found_one)
        {
            res.add(res);
        }

        if (test_bit(scalar, i))
        {
            found_one = true;
            res.add(base);
        }
    }
}

__global__
void pre_compute(affine_t* pre_points, size_t npoints) {
    const uint32_t tnum = blockDim.x * gridDim.x;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    bucket_t Pi_xyzz;
    for (uint32_t i = tid; i < npoints; i += tnum) {
        affine_t* Pi = pre_points + i;
        Pi_xyzz = *Pi;
//        for (int j = 1; j < 7; j++) {
        for (int j = 1; j < 16; j++) {
            Pi = Pi + npoints;
            Pi_xyzz.dbl();

            Pi_xyzz.xyzz_to_affine(*Pi);
        }
    }
}

__global__
void process_scalar_1(uint16_t* scalar, uint32_t* scalar_tuple,
                      uint32_t* d_scalar_map, uint32_t* point_idx, size_t npoints) {

    const uint32_t tnum = blockDim.x * gridDim.x;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < npoints; i += tnum) {
        uint16_t* cur_scalar_ptr = scalar + (i << 4);
        uint32_t cur_scalar = (uint32_t)(*cur_scalar_ptr);  // uint32_t instead of uint16_t, specifically for 0x10000
        scalar_tuple[i] = d_scalar_map[cur_scalar];

        point_idx[i] = i;

        for (int j = i + npoints; j < NWINS * npoints; j += npoints) {
            cur_scalar_ptr += 1;
            cur_scalar = (uint32_t)(*(cur_scalar_ptr));
            cur_scalar += (scalar_tuple[j - npoints] & 1);
            scalar_tuple[j] = d_scalar_map[cur_scalar];

            point_idx[j] = i;
        }
    }

}

__global__
void process_scalar_2(uint32_t* scalar_tuple_out,
                      uint16_t* bucket_idx, size_t npoints) {
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    uint16_t* bucket_idx_ptr = bucket_idx + npoints * bid;
    uint32_t* scalar_tuple_out_ptr = scalar_tuple_out + npoints * bid;

    for (uint32_t i = tid; i < npoints; i += tnum) {
        bucket_idx_ptr[i] = scalar_tuple_out_ptr[i] >> 16;
    }
}

__global__
void buffer_init(bucket_t *buckets_pre, uint16_t* bucket_idx_pre_vector, bucket_t *buckets) {  // new
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    const uint32_t buffer_len = 1 << (WBITS - 1);

    bucket_t* buckets_pre_ptr = buckets_pre + buffer_len * bid;
    uint16_t* bucket_idx_pre_vector_ptr = bucket_idx_pre_vector + buffer_len * bid;
    bucket_t* buckets_ptr = buckets + ((1 << (WBITS - 2)) + 1) * bid;

    for (uint32_t i = tid; i < buffer_len; i += tnum) {
        buckets_pre_ptr[i].inf();
        bucket_idx_pre_vector_ptr[i] = 0;
    }
    for (uint32_t i = tid; i <= 1 << (WBITS - 2); i += tnum) {
        buckets_ptr[i].inf();
    }
}

// new
__global__
void bucket_acc(uint32_t* scalar_tuple_out, uint16_t* bucket_idx, uint32_t* point_idx_out,
                affine_t* pre_points, bucket_t *buckets_pre,
                uint16_t* bucket_idx_pre_vector, size_t npoints) {  // new
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid_inner = threadIdx.x;
    const uint32_t tid = blockIdx.y * blockDim.x + tid_inner;
    const uint32_t bid = blockIdx.x;
    const uint32_t buffer_len = 1 << (WBITS - 1);

    uint32_t* scalar_tuple_out_ptr = scalar_tuple_out + npoints * bid;
    uint16_t* bucket_idx_ptr = bucket_idx + npoints * bid;
    uint32_t* point_idx_out_ptr = point_idx_out + npoints * bid;
    bucket_t* buckets_pre_ptr = buckets_pre + buffer_len * bid;
    uint16_t* bucket_idx_pre_vector_ptr = bucket_idx_pre_vector + buffer_len * bid;

    const uint32_t step_len = (npoints + tnum - 1) / tnum;

    uint32_t s = step_len * tid;
    uint32_t e = s + step_len;
    if (s >= npoints) {
        return;
    }
    if (e >= npoints) e = npoints;

    uint16_t pre_bucket_idx = 0x8000;   // not exist
    bucket_acc_smem[tid_inner * 2 + 1].inf();

    uint32_t offset = tid + ((bucket_idx_ptr[s] + 1) >> 1);
    uint32_t unique_num = 0;
    // process [s, e)
    for (uint32_t i = s; i < e; i++) {
        uint32_t power_of_2 = (scalar_tuple_out_ptr[i] >> 8) & 0x0f;

        uint16_t cur_bucket_idx = bucket_idx_ptr[i];

        if (cur_bucket_idx != pre_bucket_idx && (unique_num++)) {
            buckets_pre_ptr[offset + unique_num - 2] = bucket_acc_smem[tid_inner * 2 + 1];
            bucket_idx_pre_vector_ptr[offset + unique_num - 2] = (pre_bucket_idx + 1) >> 1;
            bucket_acc_smem[tid_inner * 2 + 1].inf();
        }
        pre_bucket_idx = cur_bucket_idx;
        bucket_acc_smem[tid_inner * 2] = pre_points[point_idx_out_ptr[i] + power_of_2 * npoints];
        if (scalar_tuple_out_ptr[i] & 0x01) {
            bucket_acc_smem[tid_inner * 2].neg(true);
        }
        bucket_acc_smem[tid_inner * 2 + 1].add(bucket_acc_smem[tid_inner * 2]);
    }
    buckets_pre_ptr[offset + unique_num - 1] = bucket_acc_smem[tid_inner * 2 + 1];
    bucket_idx_pre_vector_ptr[offset + unique_num - 1] = (pre_bucket_idx + 1) >> 1;

}

// v1.1 (2^{14} THREADS)
__global__
void bucket_acc_2(bucket_t *buckets_pre, uint16_t* bucket_idx_pre_vector, bucket_t *buckets, uint32_t* d_op_index) {
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t buffer_len = 1 << (WBITS - 1);

    bucket_t* buckets_ptr = buckets + ((1 << (WBITS - 2)) + 1) * bid; // new
    bucket_t* buckets_pre_ptr = buckets_pre + buffer_len * bid;
    uint16_t* bucket_idx_pre_vector_ptr = bucket_idx_pre_vector + buffer_len * bid;

    bucket_t inf_aux;
    inf_aux.inf();
    bucket_t* aux1[6], *aux2[6];
    uint16_t aux3[5];
    aux1[4] = &inf_aux;
    aux2[4] = &inf_aux;
    aux3[4] = 0;

    uint32_t condition, op_index, op1, op2, op3, op4, op5, op6, op7, op8;
    uint32_t k = 1;
    for (uint32_t j = buffer_len >> 2; j > NTHREADS; j >>= 1, k <<= 1) {
        for (uint32_t i = tid; i < j; i += tnum) {
            uint32_t offset = i * (k << 2);

            aux1[0] = buckets_pre_ptr + offset;
            aux1[1] = aux1[0] + k;
            aux1[2] = aux1[1] + k;
            aux1[3] = aux1[2] + k;
            aux1[5] = aux1[3];

            aux3[0] = bucket_idx_pre_vector_ptr[offset];
            aux3[1] = bucket_idx_pre_vector_ptr[offset + k];
            aux3[2] = bucket_idx_pre_vector_ptr[offset + 2 * k];
            aux3[3] = bucket_idx_pre_vector_ptr[offset + 3 * k];

            aux2[0] = buckets_ptr + aux3[0];
            aux2[1] = buckets_ptr + aux3[1];
            aux2[2] = buckets_ptr + aux3[2];
            aux2[3] = buckets_ptr + aux3[3];
            aux2[5] = aux1[2];

            condition = ((((aux3[0] == aux3[1]) || (aux3[1] == 0)) & 1) << 2)
                    | ((((aux3[1] == aux3[2]) || (aux3[2] == 0)) & 1) << 1)
                    | ((((aux3[2] == aux3[3]) || (aux3[3] == 0)) & 1));
            op_index = d_op_index[condition];
            op1 = op_index >> 28;
            op2 = (op_index >> 24) & 0x0f;
            op3 = (op_index >> 20) & 0x0f;
            op4 = (op_index >> 16) & 0x0f;
            op5 = (op_index >> 12) & 0x0f;
            op6 = (op_index >> 8) & 0x0f;
            op7 = (op_index >> 4) & 0x0f;
            op8 = (op_index) & 0x0f;

            aux1[op1]->add(*(aux1[op3]));   // OP_1
            aux3[op1] = aux3[op2] | aux3[op3];
            *(aux2[op4]) = *(aux1[op4]);    // OP_2
            aux3[2] += (aux3[3] - aux3[2]) * ((op4 == 5) & 1);
            *(aux2[op5]) = *(aux1[op5]);    // OP_3
            *(aux1[op6]) = *(aux1[op7]);    // OP_4
            aux1[op6]->add(*(aux1[op8]));
            aux3[op6] = aux3[op7] | aux3[op8];

            bucket_idx_pre_vector_ptr[offset] = aux3[0];
            bucket_idx_pre_vector_ptr[offset + 2 * k] = aux3[2];
        }
        cooperative_groups::this_grid().sync();
    }
    for (uint32_t j = NTHREADS; j > 0; j >>= 1, k <<= 1) {
        if (tid < j) {
            uint32_t offset = tid * (k << 2);

            aux1[0] = buckets_pre_ptr + offset;
            aux1[1] = aux1[0] + k;
            aux1[2] = aux1[1] + k;
            aux1[3] = aux1[2] + k;
            aux1[5] = aux1[3];

            aux3[0] = bucket_idx_pre_vector_ptr[offset];
            aux3[1] = bucket_idx_pre_vector_ptr[offset + k];
            aux3[2] = bucket_idx_pre_vector_ptr[offset + 2 * k];
            aux3[3] = bucket_idx_pre_vector_ptr[offset + 3 * k];

            aux2[0] = buckets_ptr + aux3[0];
            aux2[1] = buckets_ptr + aux3[1];
            aux2[2] = buckets_ptr + aux3[2];
            aux2[3] = buckets_ptr + aux3[3];
            aux2[5] = aux1[2];

            condition = ((((aux3[0] == aux3[1]) || (aux3[1] == 0)) & 1) << 2)
                    | ((((aux3[1] == aux3[2]) || (aux3[2] == 0)) & 1) << 1)
                    | ((((aux3[2] == aux3[3]) || (aux3[3] == 0)) & 1));
            op_index = d_op_index[condition];
            op1 = op_index >> 28;
            op2 = (op_index >> 24) & 0x0f;
            op3 = (op_index >> 20) & 0x0f;
            op4 = (op_index >> 16) & 0x0f;
            op5 = (op_index >> 12) & 0x0f;
            op6 = (op_index >> 8) & 0x0f;
            op7 = (op_index >> 4) & 0x0f;
            op8 = (op_index) & 0x0f;

            aux1[op1]->add(*(aux1[op3]));   // OP_1
            aux3[op1] = aux3[op2] | aux3[op3];
            *(aux2[op4]) = *(aux1[op4]);    // OP_2
            aux3[2] += (aux3[3] - aux3[2]) * ((op4 == 5) & 1);
            *(aux2[op5]) = *(aux1[op5]);    // OP_3
            *(aux1[op6]) = *(aux1[op7]);    // OP_4
            aux1[op6]->add(*(aux1[op8]));
            aux3[op6] = aux3[op7] | aux3[op8];

            bucket_idx_pre_vector_ptr[offset] = aux3[0];
            bucket_idx_pre_vector_ptr[offset + 2 * k] = aux3[2];
        }
        if (j > WARP_SZ) {
            cooperative_groups::this_thread_block().sync();
        }
    }
    if (tid == 0) {
        aux3[0] = bucket_idx_pre_vector_ptr[0];
        aux3[2] = bucket_idx_pre_vector_ptr[buffer_len >> 1];
        buckets_ptr[aux3[0]].add(buckets_pre_ptr[0]);
        buckets_ptr[aux3[2]].add(buckets_pre_ptr[buffer_len >> 1]);
    }

}

__global__
void bucket_agg_1(bucket_t *buckets) {
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    bucket_t* buckets_ptr = buckets + ((1 << (WBITS - 2)) + 1) * bid + 1;   // new

    for (uint32_t j = tid; j < (1 << (WBITS - 5)); j += tnum) {
        uint32_t s = j << 3;
        bucket_t* Bi = buckets_ptr + 0x3fff - s;
        for (int i = 1; i < 8; i++) {
            (Bi - i)->add(*(Bi - i + 1));
        }
    }
}

__global__
void bucket_agg_2(bucket_t *buckets) {
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    bucket_t* buckets_ptr = buckets + ((1 << (WBITS - 2)) + 1) * bid + 1;   // new

    for (uint32_t i = 3; i < 14; i++) {
        for (uint32_t k = tid; k < (1 << (WBITS - 3)); k += tnum) {
            uint32_t baseline = ((1 + (k >> i)) << (i + 1)) - (1 << i);
            uint32_t offset = k & ((1 << i) - 1);

            bucket_t* Bi = buckets_ptr + 0x3fff - (baseline - 1);
            bucket_t* Bj = Bi - (offset + 1);	// B + 0x3fff - (baseline + offset)

            Bj->add(*Bi);
        }
        cooperative_groups::this_grid().sync();
    }
}

__global__
void recursive_sum(bucket_t *buckets, bucket_t *res) {
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    bucket_t* buckets_ptr = buckets + ((1 << (WBITS - 2)) + 1) * bid + 1; // new

    if (tid == 0) {
        res[bid] = *buckets_ptr;
    }
    // cooperative_groups::this_grid().sync();

    for (uint32_t j = 1 << (WBITS - 3); j > NTHREADS; j >>= 1) {
        for (uint32_t i = tid; i < j; i += tnum) {
            buckets_ptr[i].add(buckets_ptr[i + j]);
        }
        cooperative_groups::this_grid().sync();
    }
    for (uint32_t j = NTHREADS; j > WARP_SZ; j >>= 1) {
        if (tid < j) {
            buckets_ptr[tid].add(buckets_ptr[tid + j]);
        }
        cooperative_groups::this_thread_block().sync();
    }

    if (tid < WARP_SZ) {
        buckets_ptr[tid].add(buckets_ptr[tid + 32]);
        buckets_ptr[tid].add(buckets_ptr[tid + 16]);
        buckets_ptr[tid].add(buckets_ptr[tid + 8]);
        buckets_ptr[tid].add(buckets_ptr[tid + 4]);
        buckets_ptr[tid].add(buckets_ptr[tid + 2]);
        buckets_ptr[tid].add(buckets_ptr[tid + 1]);
    }
    if (tid == 0) {
        buckets_ptr->dbl();
        res[bid].neg(true);
        res[bid].add(*buckets_ptr);
    }

    /*cooperative_groups::this_grid().sync();
    if (tid == 0 && bid == 0) {
    bucket_t check_res;
    check_res.inf();

    for (int i = 15; i > -1; i--) {
	for (int j = 0; j < 16; j++) {
	    check_res.add(check_res);
	}
	check_res.add(res[i]);
    }
    printf("\ncheck_2:\n");
    check_res.xyzz_print();
    }*/
}

#else

#include <cassert>
#include <vector>
using namespace std;

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>
#include <util/host_pinned_allocator_t.hpp>


template<typename... Types>
inline void launch_coop(void(*f)(Types...),
                        dim3 gridDim, dim3 blockDim, cudaStream_t stream,
                        Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                        va_args, 0, stream));
}

class stream_t {
    cudaStream_t stream;
public:
    stream_t(int device)  {
        CUDA_OK(cudaSetDevice(device));
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    ~stream_t() { cudaStreamDestroy(stream); }
    inline operator decltype(stream)() { return stream; }
};


template<class bucket_t> class result_t_faster {
    bucket_t ret[NWINS];
public:
    result_t_faster() {}
    inline operator decltype(ret)&() { return ret; }
};

template<class T>
class device_ptr_list_t {
    vector<T*> d_ptrs;
public:
    device_ptr_list_t() {}
    ~device_ptr_list_t() {
        for(T *ptr: d_ptrs) {
            cudaFree(ptr);
        }
    }
    size_t allocate(size_t bytes) {
        T *d_ptr;
        CUDA_OK(cudaMalloc(&d_ptr, bytes));
        d_ptrs.push_back(d_ptr);
        return d_ptrs.size() - 1;
    }
    size_t size() {
        return d_ptrs.size();
    }
    T* operator[](size_t i) {
        if (i > d_ptrs.size() - 1) {
            CUDA_OK(cudaErrorInvalidDevicePointer);
        }
        return d_ptrs[i];
    }

};

// Pippenger MSM class
template<class bucket_t, class point_t, class affine_t, class scalar_t>
class pippenger_t {
public:
    typedef vector<result_t_faster<bucket_t>,
                   host_pinned_allocator_t<result_t_faster<bucket_t>>> result_container_t_faster;

private:
    size_t sm_count;
    bool init_done = false;
    device_ptr_list_t<affine_t> d_base_ptrs;
    device_ptr_list_t<scalar_t> d_scalar_ptrs;
    device_ptr_list_t<bucket_t> d_bucket_ptrs;

    device_ptr_list_t<bucket_t> d_bucket_pre_ptrs;  // v1.1
    device_ptr_list_t<uint16_t> d_bucket_idx_pre_ptrs;  // v1.1

    device_ptr_list_t<bucket_t> d_res_ptrs;

    // GPU device number
    int device;

    // TODO: Move to device class eventually
    thread_pool_t *da_pool = nullptr;

public:
    // Default stream for operations
    stream_t default_stream;

    device_ptr_list_t<uint32_t> d_op_index; // new
    device_ptr_list_t<uint32_t> d_scalar_map;
    device_ptr_list_t<uint32_t> d_scalar_tuple_ptrs;
    device_ptr_list_t<uint32_t> d_point_idx_ptrs;
    device_ptr_list_t<uint16_t> d_bucket_idx_ptrs;
    device_ptr_list_t<unsigned char> d_cub_ptrs;

    // Parameters for an MSM operation
    class MSMConfig {
        friend pippenger_t;
    public:
        size_t npoints;
        size_t N;
        size_t n;
    };

    pippenger_t() : default_stream(0) {
        device = 0;
    }

    pippenger_t(int _device, thread_pool_t *pool = nullptr)
        : default_stream(_device) {
        da_pool = pool;
        device = _device;
    }

    // Initialize instance. Throws cuda_error on error.
    void init() {
        if (!init_done) {
            CUDA_OK(cudaSetDevice(device));
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 7)
                CUDA_OK(cudaErrorInvalidDevice);
            sm_count = prop.multiProcessorCount;

            if (da_pool == nullptr) {
                da_pool = new thread_pool_t();
            }

            init_done = true;
        }
    }

    int get_device() {
        return device;
    }

    // Initialize parameters for a specific size MSM. Throws cuda_error on error.
    MSMConfig init_msm_faster(size_t npoints) {
        init();

        MSMConfig config;
        config.npoints = npoints;
        config.n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);
        config.N = (sm_count*256) / (NTHREADS*NWINS);
        size_t delta = ((npoints+(config.N)-1)/(config.N)+WARP_SZ-1) & (0U-WARP_SZ);
        config.N = (npoints+delta-1) / delta;

//        if(config.N % 2 == 1) config.N -= 1;
        return config;
    }

    size_t get_size_bases(MSMConfig& config) {
        return config.n * sizeof(affine_t);
    }
    size_t get_size_scalars(MSMConfig& config) {
        return config.n * sizeof(scalar_t);
    }
    size_t get_size_buckets() { // new
        return sizeof(bucket_t) * NWINS * ((1 << (WBITS - 2)) + 1);
    }
    size_t get_size_buckets_pre() {    // new
        return sizeof(bucket_t) * NWINS * (1 << (WBITS - 1));   // redundant
    }
    size_t get_size_bucket_idx_pre_vector() {  // new
        return sizeof(uint16_t) * NWINS * (1 << (WBITS - 1));
    }
    size_t get_size_res() {
        return sizeof(bucket_t) * NWINS;
    }

    size_t get_size_scalar_map() {
        return ((1 << 16) + 1) * sizeof(uint32_t);
    }
    size_t get_size_scalar_tuple(MSMConfig& config) {
        return config.n * sizeof(uint32_t) * NWINS;
    }
    size_t get_size_point_idx(MSMConfig& config) {
        return config.n * sizeof(uint32_t) * NWINS;
    }
    size_t get_size_bucket_idx(MSMConfig& config) {
        return config.n * sizeof(uint16_t) * NWINS;
    }

    size_t get_size_cub_sort_faster(MSMConfig& config){
        uint32_t *d_scalar_tuple = nullptr;
        uint32_t *d_scalar_tuple_out = nullptr;
        uint32_t *d_point_idx = nullptr;
        uint32_t *d_point_idx_out = nullptr;
        void *d_temp = NULL;
        size_t temp_size = 0;
        cub::DeviceRadixSort::SortPairs(d_temp, temp_size,
                                        d_scalar_tuple, d_scalar_tuple_out,
                                        d_point_idx, d_point_idx_out, config.n, 0, 31);
        return temp_size;
    }

    result_container_t_faster get_result_container_faster() {
        result_container_t_faster res(1);
        return res;
    }

    // Allocate storage for bases on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_bases(MSMConfig& config) {
//        return d_base_ptrs.allocate(7 * get_size_bases(config));
        return d_base_ptrs.allocate(16 * get_size_bases(config));
    }

    size_t allocate_d_scalars(MSMConfig& config) {
        return d_scalar_ptrs.allocate(get_size_scalars(config));
    }

    size_t allocate_d_buckets() {
        return d_bucket_ptrs.allocate(get_size_buckets());
    }
    size_t allocate_d_buckets_pre() {  // v1.1
        return d_bucket_pre_ptrs.allocate(get_size_buckets_pre());
    }
    size_t allocate_d_bucket_idx_pre_vector() {  // v1.1
        return d_bucket_idx_pre_ptrs.allocate(get_size_bucket_idx_pre_vector());
    }

    size_t allocate_d_res() {
        return d_res_ptrs.allocate(get_size_res());
    }

    size_t allocate_d_scalar_map() {
        return d_scalar_map.allocate(get_size_scalar_map());
    }

    size_t allocate_d_op_index() {  // new
        return d_op_index.allocate(8 * sizeof(uint32_t));
    }

    size_t allocate_d_scalar_tuple(MSMConfig& config) {
        return d_scalar_tuple_ptrs.allocate(get_size_scalar_tuple(config));
    }
    size_t allocate_d_scalar_tuple_out(MSMConfig& config) {
        return d_scalar_tuple_ptrs.allocate(get_size_scalar_tuple(config));
    }

    size_t allocate_d_point_idx(MSMConfig& config) {
        return d_point_idx_ptrs.allocate(get_size_point_idx(config));
//        return d_point_idx_ptrs.allocate(config.n * sizeof(uint32_t));
    }
    size_t allocate_d_point_idx_out(MSMConfig& config) {
        return d_point_idx_ptrs.allocate(get_size_point_idx(config));
    }

    size_t allocate_d_bucket_idx(MSMConfig& config) {
        return d_bucket_idx_ptrs.allocate(get_size_bucket_idx(config));
    }

    size_t allocate_d_cub_sort_faster(MSMConfig& config) {
        return d_cub_ptrs.allocate(get_size_cub_sort_faster(config));
    }

    // Transfer bases to device. Throws cuda_error on error.
    void transfer_bases_to_device(MSMConfig& config, size_t d_bases_idx, const affine_t points[],
                                  size_t ffi_affine_sz = sizeof(affine_t),
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        CUDA_OK(cudaSetDevice(device));
        if (ffi_affine_sz != sizeof(*d_points))
            CUDA_OK(cudaMemcpy2DAsync(d_points, sizeof(*d_points),
                                      points, ffi_affine_sz,
                                      ffi_affine_sz, config.npoints,
                                      cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpyAsync(d_points, points, config.npoints*sizeof(*d_points),
                                    cudaMemcpyHostToDevice, stream));
    }

    // Transfer scalars to device. Throws cuda_error on error.
    void transfer_scalars_to_device(MSMConfig& config,
                                    size_t d_scalars_idx, const scalar_t scalars[],
                                    cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(d_scalars, scalars, config.npoints*sizeof(*d_scalars),
                                cudaMemcpyHostToDevice, stream));
    }


    void transfer_res_to_host_faster(result_container_t_faster &res, size_t d_res_idx,
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t *d_res = d_res_ptrs[d_res_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(res[0], d_res, sizeof(res[0]),
                                cudaMemcpyDeviceToHost, stream));
    }

    void transfer_scalar_map_to_device(size_t d_scalar_map_idx, const uint32_t scalar_map[],
                                       cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        uint32_t *d_smap = d_scalar_map[d_scalar_map_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(d_smap, scalar_map, ((1 << 16) + 1)*sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));
    }

    void transfer_op_index_to_device(size_t d_op_index_sn, const uint32_t op_index[],
                                       cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        uint32_t *d_op_idx = d_op_index[d_op_index_sn];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(d_op_idx, op_index, 8*sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));
    }

    void synchronize_stream() {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamSynchronize(default_stream));
    }

    void launch_kernel_init(MSMConfig& config,
                            size_t d_points_sn, cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_points_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(pre_compute, NWINS * config.N, NTHREADS, stream,
                    d_points, config.npoints);
    }

    void launch_process_scalar_1(MSMConfig& config,
                                 size_t d_scalars_sn, size_t d_scalar_tuples_sn,
                                 size_t d_scalar_map_sn, size_t d_point_idx_sn,
                                 cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        uint16_t* d_scalars = (uint16_t*)d_scalar_ptrs[d_scalars_sn];
        uint32_t* d_scalar_tuple = d_scalar_tuple_ptrs[d_scalar_tuples_sn];
        uint32_t* d_smap = d_scalar_map[d_scalar_map_sn];
        uint32_t* d_point_idx = d_point_idx_ptrs[d_point_idx_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(process_scalar_1, NWINS * config.N, NTHREADS, stream,
                    d_scalars, d_scalar_tuple, d_smap, d_point_idx, config.npoints);
    }

    void launch_process_scalar_2(MSMConfig& config,
                                 size_t d_scalar_tuples_out_sn, size_t d_bucket_idx_sn,
                                 cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        uint32_t* d_scalar_tuple_out = d_scalar_tuple_ptrs[d_scalar_tuples_out_sn];
        uint16_t* d_bucket_idx = d_bucket_idx_ptrs[d_bucket_idx_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(process_scalar_2, dim3(NWINS, config.N), NTHREADS, stream,
                    d_scalar_tuple_out, d_bucket_idx, config.npoints);
    }

    void launch_buffer_init(MSMConfig& config, size_t d_buckets_pre_sn, size_t d_bucket_idx_pre_vector_sn,
                            size_t d_buckets_sn, cudaStream_t s = nullptr) { // new
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t* d_buckets_pre = d_bucket_pre_ptrs[d_buckets_pre_sn];
        uint16_t* d_bucket_idx_pre_vector = d_bucket_idx_pre_ptrs[d_bucket_idx_pre_vector_sn];
        bucket_t* d_buckets = d_bucket_ptrs[d_buckets_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(buffer_init, dim3(NWINS, config.N), NTHREADS, stream,
                    d_buckets_pre, d_bucket_idx_pre_vector, d_buckets);
    }

    void launch_bucket_acc(MSMConfig& config,
                           size_t d_scalar_tuples_out_sn, size_t d_bucket_idx_sn,
                           size_t d_point_idx_out_sn, size_t d_points_sn, size_t d_buckets_sn,
                           size_t d_buckets_pre_sn, size_t d_bucket_idx_pre_vector_sn,
                           size_t d_op_index_sn, cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        uint32_t* d_scalar_tuple_out = d_scalar_tuple_ptrs[d_scalar_tuples_out_sn];
        uint16_t* d_bucket_idx = d_bucket_idx_ptrs[d_bucket_idx_sn];
        uint32_t* d_point_idx_out = d_point_idx_ptrs[d_point_idx_out_sn];
        affine_t* d_points = d_base_ptrs[d_points_sn];
        bucket_t* d_buckets = d_bucket_ptrs[d_buckets_sn];
        bucket_t* d_buckets_pre = d_bucket_pre_ptrs[d_buckets_pre_sn];
        uint16_t* d_bucket_idx_pre_vector = d_bucket_idx_pre_ptrs[d_bucket_idx_pre_vector_sn];
        uint32_t* d_op_idx = d_op_index[d_op_index_sn];

        CUDA_OK(cudaSetDevice(device));

        launch_coop(bucket_acc, dim3(NWINS, config.N), NTHREADS, stream,
                    d_scalar_tuple_out, d_bucket_idx, d_point_idx_out,
                    d_points, d_buckets_pre,
                    d_bucket_idx_pre_vector, config.npoints);
        launch_coop(bucket_acc_2, dim3(NWINS, config.N), NTHREADS, stream,
                    d_buckets_pre, d_bucket_idx_pre_vector, d_buckets, d_op_idx);
    }

    void launch_bucket_agg_1(MSMConfig& config, size_t d_buckets_sn, cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t* d_buckets = d_bucket_ptrs[d_buckets_sn];

        CUDA_OK(cudaSetDevice(device));
//        bucket_agg_1<<<dim3(NWINS, (1 << (WBITS - 5)) / NTHREADS), NTHREADS, 0, stream>>>(d_buckets);
        launch_coop(bucket_agg_1, dim3(NWINS, config.N), NTHREADS, stream, d_buckets);
    }

    void launch_bucket_agg_2(MSMConfig& config, size_t d_buckets_sn, cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t* d_buckets = d_bucket_ptrs[d_buckets_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(bucket_agg_2, dim3(NWINS, config.N), NTHREADS, stream, d_buckets);
    }

    void launch_recursive_sum(MSMConfig& config, size_t d_buckets_sn, size_t d_res_sn, cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t* d_buckets = d_bucket_ptrs[d_buckets_sn];
        bucket_t* d_res = d_res_ptrs[d_res_sn];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(recursive_sum, dim3(NWINS, config.N), NTHREADS, stream, d_buckets, d_res);
    }

    // Perform final accumulation on CPU.
    void accumulate_faster(point_t &out, result_container_t_faster &res) {
        out.inf();

        for(int32_t k = NWINS - 1; k >= 0; k--)
        {
            for (int32_t i = 0; i < WBITS; i++)
            {
                out.dbl();
            }
            point_t p = (res[0])[k];
            out.add(p);
        }

    }
};

#endif
