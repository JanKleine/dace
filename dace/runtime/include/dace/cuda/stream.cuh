#ifndef __DACE_STREAM_CUH
#define __DACE_STREAM_CUH

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <new> // Used for the in-memory ctor call in the move assignment operator below  

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../../../../external/cub/cub/util_ptx.cuh"
#include "../../../../external/cub/cub/warp/warp_reduce.cuh"
#include "../../../../external/cub/cub/warp/warp_scan.cuh"

#include "cudacommon.cuh"

namespace dace {
    // Adapted from https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
    __inline__ __device__ uint32_t atomicAggInc(uint32_t *ctr) {
        auto g = cooperative_groups::coalesced_threads();
        uint32_t warp_res;
        int rank = g.thread_rank();
        if (rank == 0)
            warp_res = atomicAdd(ctr, g.size());
        return g.shfl(warp_res, 0) + rank;
    }

    __inline__ __device__ uint32_t atomicAggDec(uint32_t *ctr) {
        auto g = cooperative_groups::coalesced_threads();
        uint32_t warp_res;
        int rank = g.thread_rank();
        if (rank == 0)
            warp_res = atomicAdd(ctr, -g.size());
        return g.shfl(warp_res, 0) + rank;
    }

    /*
    __inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
        for (int offset = CUB_PTX_WARP_THREADS / 2; offset > 0; offset /= 2)
            val += __shfl_down(val, offset);
        return val;
    }
    */

    //
    // Queue classes (device):
    //

    /*
    * @brief A device-level MPMC Queue
    */
    template<typename T, bool IS_POWEROFTWO = false>
    class GPUStream
    {
    public:
        T* m_data;
        uint32_t *m_start, *m_end, *m_pending;
        uint32_t m_capacity_mask;

        __host__ GPUStream() : m_data(nullptr), m_start(nullptr), m_end(nullptr),
            m_pending(nullptr), m_capacity_mask(0) {}
        __host__ __device__ GPUStream(T* data, uint32_t capacity,
                                      uint32_t *start, uint32_t *end, 
                                      uint32_t *pending) :
            m_data(data), m_start(start), m_end(end), m_pending(pending),
            m_capacity_mask(IS_POWEROFTWO ? (capacity - 1) : capacity)
        {
            if (IS_POWEROFTWO) {
                assert((capacity - 1 & capacity) == 0); // Must be a power of two for handling circular overflow correctly  
            }
        }

        __device__ __forceinline__ void reset() const
        {
            *m_start = 0; 
            *m_end = 0;
            *m_pending = 0;
        }

        __device__ __forceinline__ T pop()
        {
            uint32_t allocation = atomicAggInc(m_start);
            return m_data[get_addr(allocation)];
        }

        /**
         * pop try
         * 
         * Pops an element from the stream as long as enough elements are
         * available with warp level aggregation. All threads that get an
         * element return 1, and place the popped element into `element`.
         * `element` will be overwritten, but CANNOT be used when 0 is returned!
         * 
         * @param element reference to where the element is to be copied
         * @return 1 iff element was popped for the calling thread, 0 otherwise.
         */
        // TODO add variant with chunk size (returns number of popped elements)
        __device__ __forceinline__ int pop_try(T &element){

            auto g = cooperative_groups::coalesced_threads();
            int rank = g.thread_rank();
            int size = g.size();

            uint32_t old_start, new_start, assumed;

            if (rank == 0) {
                old_start = *m_start;
                do {
                    assumed = old_start;
                    new_start = min(*m_end, assumed + size);
                    old_start = atomicCAS(m_start, assumed, new_start);
                } while (assumed != old_start);
            }

            new_start = g.shfl(new_start, 0);
            old_start = g.shfl(old_start, 0);

            // always fetch element to reduce warp divergence
            element = m_data[get_addr(old_start + rank)];
            return rank < (new_start - old_start);

        }

        // all threads are expected to call with the same count variable!
        __device__ __forceinline__ int pop_soft(T* elements, uint32_t count){

            auto g = cooperative_groups::coalesced_threads();
            int rank = g.thread_rank();
            int size = g.size();
            int max_elements = size * count;

            uint32_t old_start, new_start, assumed;

            if (rank == 0) {
                old_start = *m_start;
                do {
                    assumed = old_start;
                    new_start = min(*m_end, assumed + max_elements);
                    old_start = atomicCAS(m_start, assumed, new_start);
                } while (assumed != old_start);
            }

            new_start = g.shfl(new_start, 0);
            old_start = g.shfl(old_start, 0);

            int num_elements = (new_start - old_start) / size;
            int reset = (new_start - old_start) % size;
            int offset = num_elements * rank + old_start;
            offset += ((rank < reset) ? rank : reset);
            num_elements += (rank < reset);

            for (int i = 0; i < num_elements; i++) {
                elements[i] = m_data[get_addr(offset + i)];
            }
            return num_elements;
        }

        __device__ __forceinline__ T *leader_pop(uint32_t count) {
            uint32_t current = *m_start;
            T *result = m_data + get_addr(current);
            *m_start += count;
            return result;
        }


        __device__ __forceinline__ uint32_t get_addr(const uint32_t& i) const {
            if (IS_POWEROFTWO)
                return i & m_capacity_mask;
            else
                return i % m_capacity_mask;
        }

        __device__ __forceinline__ void push(const T& item)
        {
            uint32_t allocation = atomicAggInc(m_pending);
            m_data[get_addr(allocation)] = item;
        }

        /*
        __device__ __forceinline__ void push(T *items, int count) 
        {
            // Perform a warp-wide scan to get thread offsets
            typedef cub::WarpScan<int> WarpScan;
            __shared__ typename WarpScan::TempStorage temp_storage[4];
            int offset;
            int warp_id = threadIdx.x / 32;
            WarpScan(temp_storage[warp_id]).ExclusiveSum(count, offset);

            // Atomic-add the total count once per warp
            uint32_t addr;
            if (threadIdx.x & 31 == 31) // Last thread
                addr = atomicAdd(m_pending, offset + count);
            // Broadcast starting address
            addr = cub::ShuffleIndex(addr, 31, 0xffffffff);

            // Copy data from each thread
            for(int i = 0; i < count; ++i)
                m_data[get_addr(addr + offset + i)] = items[i];
        }
        */

        __device__ __forceinline__ void prepend(const T& item)
        {
            uint32_t allocation = atomicAggDec(m_start) - 1;
            m_data[get_addr(allocation)] = item;
        }

        __device__ __forceinline__ T read(uint32_t i) const
        {
            return m_data[get_addr(*m_start + i)];
        }
                        
        __device__ __forceinline__ uint32_t count() const
        {
            return *m_end - *m_start;
        }

        /**
         * commit_pending
         *
         * Commit all pending elements (added with `push()`) to stream. This
         * should be executed by as few threads as possible (for example only
         * one thread per thread block).
         *
         * @return number of committed elements
         */
        __device__ __forceinline__ uint32_t commit_pending() const
        {
            uint32_t assumed, old_end = *m_end, new_end;

            do {
                assumed = old_end;
                new_end = *m_pending;
                old_end = atomicCAS(m_end, assumed, new_end);
            } while (assumed != old_end);

            return new_end - old_end;
        }

        __device__ __forceinline__ uint32_t get_start() const
        {
            return *m_start;
        }

        __device__ __forceinline__ uint32_t get_start_delta(uint32_t prev_start) const
        {
            return prev_start - *m_start;
        }
    };

    template <int CHUNKSIZE>
    struct GPUConsume {
        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename Functor>
        __device__ __forceinline__ static void consume(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                            Functor&& contents) {
            assert(false);
        }

        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename CondFunctor, typename Functor>
        __device__ __forceinline__ static void consume_cond(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                                 CondFunctor&& quiescence, Functor&& contents) {
            assert(false);
        }
    };

    // Specialization for consumption of 1 element
    template<>
    struct GPUConsume<1> {
        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED, typename Functor>
        __device__ __forceinline__ static void consume(cub::GridBarrier __gbar, StreamT<T, ALIGNED>& stream, unsigned num_threads, Functor&& contents) {

            // continue flag in case of tbd map
            cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
            int rank = block.thread_rank();
            int idx = rank + (blockDim.x * blockIdx.x);

            if (idx == 0) stream.commit_pending();
            __gbar.Sync();

            T element;
            while (stream.count()) {
                // all threads enter without changing the value
                __gbar.Sync();
                uint32_t popped = stream.pop_try(element);
                // TODO do thread block thing

                if (popped) {
                    contents(idx, element);
                }

                block.sync();
                if (rank == 0) {
                    stream.commit_pending();
                }

                // wait for all thread to finish so the count value will be accurate
                __gbar.Sync();
            }
        }

        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename CondFunctor, typename Functor>
        __device__ __forceinline__ static void consume_cond(StreamT<T, ALIGNED>& stream, unsigned num_threads,
                                 CondFunctor&& quiescence, Functor&& contents) {
            assert(false);
        }
    };

    ////////////////////////////////////////////////////////////
    // Host controllers for GPU streams

    template<typename T, bool IS_POW2>
    __global__ void ResetGPUStream_kernel(GPUStream<T, IS_POW2> stream)
    {
        stream.reset();
    }

    template<typename T, bool IS_POW2>
    void ResetGPUStream(GPUStream<T, IS_POW2>& stream)
    {
        void *args_reset[1] = { &stream };
        DACE_CUDA_CHECK(cudaLaunchKernel((void *)&ResetGPUStream_kernel<T, IS_POW2>,
                                         dim3(1, 1, 1), dim3(1, 1, 1), 
                                         args_reset, 0, (cudaStream_t)0));
    }

    template<typename T, bool IS_POW2>
    __global__ void PushToGPUStream_kernel(GPUStream<T, IS_POW2> stream, T item)
    {
        stream.push(item);
        stream.commit_pending();
    }

    template<typename T, bool IS_POW2>
    void PushToGPUStream(GPUStream<T, IS_POW2>& stream, const T& item)
    {
        void *args_push[2] = { &stream, &item };
        DACE_CUDA_CHECK(cudaLaunchKernel((void *)&PushToGPUStream_kernel<T, IS_POW2>,
                                         dim3(1, 1, 1), dim3(1, 1, 1), 
                                         args_push, 0, (cudaStream_t)0));
    }

    ////////////////////////////////////////////////////////////
    // Host memory management for GPU streams


    template<typename T, bool IS_POW2>
    GPUStream<T, IS_POW2> AllocGPUArrayStreamView(T *ptr, uint32_t capacity)
    {
        uint32_t *gStart, *gEnd, *gPending;
        DACE_CUDA_CHECK(cudaMalloc(&gStart, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMalloc(&gEnd, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMalloc(&gPending, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemset(gStart, 0, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemset(gEnd, 0, sizeof(uint32_t)));
        DACE_CUDA_CHECK(cudaMemset(gPending, 0, sizeof(uint32_t)));
        return GPUStream<T, IS_POW2>(ptr, capacity, gStart, gEnd, gPending);
    }

    template<typename T, bool IS_POW2>
    GPUStream<T, IS_POW2> AllocGPUStream(uint32_t capacity)
    {
        T *gData;
        DACE_CUDA_CHECK(cudaMalloc(&gData, capacity * sizeof(T)));
        return AllocGPUArrayStreamView<T, IS_POW2>(gData, capacity);
    }

    template<typename T, bool IS_POW2>
    void FreeGPUArrayStreamView(GPUStream<T, IS_POW2>& stream)
    {
        DACE_CUDA_CHECK(cudaFree(stream.m_start));
        DACE_CUDA_CHECK(cudaFree(stream.m_end));
        DACE_CUDA_CHECK(cudaFree(stream.m_pending));
    }

    template<typename T, bool IS_POW2>
    void FreeGPUStream(GPUStream<T, IS_POW2>& stream)
    {
        FreeGPUArrayStreamView(stream);
        DACE_CUDA_CHECK(cudaFree(stream.m_data));
    }

}  // namespace dace
#endif // __DACE_STREAM_CUH