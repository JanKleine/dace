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

namespace cg = cooperative_groups;

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
        volatile uint32_t *m_start, *m_end, *m_pending;
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
            uint32_t allocation = atomicAggInc((uint32_t *) m_start);
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

            auto g = cg::coalesced_threads();
            int rank = g.thread_rank();
            int size = g.size();

            uint32_t old_start, new_start, assumed;

            if (rank == 0) {
                old_start = *m_start;
                do {
                    assumed = old_start;
                    new_start = min(*m_end, assumed + size);
                    old_start = atomicCAS((uint32_t *) m_start, assumed, new_start);
                } while (assumed != old_start);
                // printf("Thread %d of block %d managed to get %d items for %d threads\n", threadIdx.x, blockIdx.x, (new_start - old_start), size);
            }
            // g.sync();

            new_start = g.shfl(new_start, 0);
            old_start = g.shfl(old_start, 0);

            // always fetch element to reduce warp divergence
            element = m_data[get_addr(old_start + rank)];
            return rank < (new_start - old_start);

        }

        // all threads are expected to call with the same count variable!
        __device__ __forceinline__ int pop_try(T* elements, uint32_t count){

            auto g = cg::coalesced_threads();
            int rank = g.thread_rank();
            int size = g.size();
            int max_elements = size * count;

            uint32_t old_start, new_start, assumed;

            if (rank == 0) {
                old_start = *m_start;
                do {
                    assumed = old_start;
                    new_start = min(*m_end, assumed + max_elements);
                    old_start = atomicCAS((uint32_t *) m_start, assumed, new_start);
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
            uint32_t allocation = atomicAggInc((uint32_t *) m_pending);
            m_data[get_addr(allocation)] = item;
            printf("Thread %d of block %d pushed node %u with depth %u to stream\n", threadIdx.x, blockIdx.x, item.vertex, item.depth);
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
            uint32_t allocation = atomicAggDec((uint32_t *) m_start) - 1;
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
            // printf("Thread %d of block %d commiting %u pendings\n", threadIdx.x, blockIdx.x, (*m_pending - *m_end));
            // __threadfence();
            do {
                assumed = old_end;
                new_end = *m_pending;
                old_end = atomicCAS((uint32_t *) m_end, assumed, new_end);
            } while (assumed != old_end);

            printf("element 0 after commiting is (%u, %u)\n", read(0).vertex, read(0).depth);

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

    template<typename T, bool IS_POWEROFTWO = false>
    class GPUStreamBroker
    {
    public:
        T* m_data;
        uint32_t *m_head, *m_tail, *m_ticket;
        int32_t *m_count;
        uint32_t m_capacity_mask, m_capacity, max_threads = 128; // TODO remove hardcode

        __host__ __device__ GPUStreamBroker(T* data, uint32_t *ticket, uint32_t capacity,
                                      uint32_t *head, uint32_t *tail, uint32_t *count) :
                                      m_data(data), m_ticket(ticket), m_head(head), m_tail(tail),
                                      m_count((int32_t *) count),
                                      m_capacity_mask(IS_POWEROFTWO ? (capacity - 1) : capacity),
                                      m_capacity(capacity)
        {
            if (IS_POWEROFTWO) {
                assert((capacity - 1 & capacity) == 0); // Must be a power of two for handling circular overflow correctly  
            }
        }

        __device__ __forceinline__ void reset() {
            *m_head = 0; 
            *m_tail = 0;
            *m_count = 0;
            for (int i = 0; i < m_capacity; i++) {
                m_ticket[i] = 0;
            }
        }

        __device__ __forceinline__ bool push(T& item) {
            while (!ensureEnqueue()) {
                uint32_t head = *m_head;
                uint32_t tail = *m_tail;
                if (m_capacity <= tail - head && tail - head < m_capacity + max_threads/2) {
                    return false;
                }
            }
            putData(item);
            return true;
        }

        __device__ __forceinline__ bool ensureEnqueue () {
            uint32_t num = *m_count;
            while (true) {
                if (num >= m_capacity) {
                    return false;
                }
                if (atomicAdd(m_count, 1) < m_capacity) {
                    return true;
                }
                num = atomicAdd(m_count, -1) - 1;
            }
        }

        __device__ __forceinline__ bool putData(T& item) {
            uint32_t pos = atomicAdd(m_tail, 1);
            uint32_t p = get_addr(pos);
            waitForTicket(p, 2 * (pos / m_capacity));
            m_data[p] = item;
            m_ticket[p] = 2 * (pos / m_capacity) + 1;
        }

        __device__ __forceinline__ bool pop(T& item) {
            while (!ensureDequeue()) {
                uint32_t head = *m_head;
                uint32_t tail = *m_tail;
                if (m_capacity + max_threads / 2 <= tail - head - 1) {
                    return false; 
                }
            }
            item = readData();
            return true;
        }

        __device__ __forceinline__ bool ensureDequeue() {
            uint32_t num = *m_count;
            while (true) {
                if (num <= 0) {
                    return false;
                }
                if (atomicAdd(m_count, -1) > 0) {
                    return true;
                }
                num = atomicAdd(m_count, 1) + 1;
            }
        }

        __device__ __forceinline__ T readData() {
            uint32_t pos = atomicAdd(m_head, 1);
            uint32_t p = get_addr(pos);
            waitForTicket(p,  2 * (pos / m_capacity) + 1);
            T item = m_data[p];
            m_ticket[p] = 2 * ((pos + m_capacity)/ m_capacity);
            return item;
        }

        __device__ __forceinline__ int waitForTicket(uint32_t pos, uint32_t expected) {
            uint32_t ticket = m_ticket[pos];
            uint32_t ns = 8;
            while (ticket != expected) {
                __nanosleep(ns); //TODO remove because need compute capability >= 7.0
                ticket = m_ticket[pos];
                if (ns < 256) {
                    ns *= 2;
                }
            }
        }

        __device__ __forceinline__ uint32_t get_addr(const uint32_t& i) const {
            if (IS_POWEROFTWO)
                return i & m_capacity_mask;
            else
                return i % m_capacity_mask;
        }

        __device__ __forceinline__ uint32_t count() {
            return *m_head - *m_tail;
        }
    };

__device__ struct GPUConsumeData_t {
    volatile int32_t waiting; // #blocks currently waiting for work in wait loop
    volatile int32_t terminate; // if true all threads will termiante the consume
    volatile int32_t terminate_vote; // if != 0 blocks will vote weather to terminate
    volatile int32_t termiante_count; // number of blocks agreeing to termiante
} global;

    template <int CHUNKSIZE>
    struct GPUConsume {
        template <template <typename, bool> typename StreamT, typename T, bool ALIGNED,
                  typename Functor>
        __device__ __forceinline__ static void consume(
                StreamT<T, ALIGNED>& stream,
                unsigned num_threads,
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

        template <template <typename, bool> typename StreamT,
                  typename T,
                  bool ALIGNED,
                  typename Functor>
        __device__ __forceinline__ static void consume(
                cub::GridBarrier __gbar,
                StreamT<T, ALIGNED>& stream,
                uint32_t num_threads,
                Functor&& contents) {
            
            // init local values
            uint32_t l_got_work = 0;
            T work_item;

            cg::thread_block block = cg::this_thread_block();
            int rank = block.thread_rank();
            int idx = (blockDim.x * blockIdx.x) + rank;
            int num_blocks = gridDim.x;

            // init block values
            __shared__ volatile uint32_t s_got_work, s_final_try;
            if (rank == 0) {
                s_got_work = 0;
                s_final_try = 0;
            }

            // init global values
            if (idx == 0) {
                global.waiting = 0;
                global.terminate = 0;
                global.terminate_vote = 0;
                global.termiante_count = 0;
                stream.commit_pending();
                // printf("%d elements in queue\n", stream.count());
            }

            if (rank == 0) printf("Block %d entered consume\n", blockIdx.x);

            __gbar.Sync();

            // main loop, only left when consume terminates
            while (!global.terminate) {
                // enter waiting state
                if (rank == 0) {
                    s_got_work = 0;
                    s_final_try = 0;
                    atomicAdd((int32_t *) &global.waiting, 1);
                }
                block.sync();

                // wait for work until termination vode is held
                while (!s_got_work && !global.terminate_vote) {

                    // if (rank ==0) printf("Block %d trying to get work\n", blockIdx.x);
                    if (rank == 0 && global.waiting == num_blocks) {
                        // printf("Block %d entering final try (%d)\n", blockIdx.x, global.waiting);
                        s_final_try += 1;
                    }

                    // attempt to pop work from stream if count is > 0
                    l_got_work = stream.pop_try(work_item);
                    // printf("Thread %d of block %d (%d)\n", rank, blockIdx.x, l_got_work);
                    if (l_got_work) {
                        printf("Thread %d of block %d got work (vertex=%u, depth=%d)\n", rank, blockIdx.x, work_item.vertex, work_item.depth);
                        s_got_work = 1;
                    }
                    block.sync();
                    if (rank == 0 && s_got_work) printf("Block %d got work\n", blockIdx.x);

                    if (rank == 0 && !s_got_work) {
                        printf("Block %d got no work, retrying...\n", blockIdx.x);
                    }

                    if (rank == 0 && s_final_try >= 3) {
                        // no working threads && no work => probably terminate
                        if (!s_got_work) {
                            // printf("Block %d called for termination vote\n", blockIdx.x);
                            global.terminate_vote = 1;
                        } else {
                            s_final_try = 0;
                        }
                    }
                    block.sync();

                }

                // leaving waiting state
                if (rank == 0) {
                    atomicAdd((int32_t *) &global.waiting, -1);
                }

                // execute termination vote
                if (global.terminate_vote) {
                    // if (rank == 0) printf("Block %d entered termination vote\n", blockIdx.x);
                    // blocks without work second the termination
                    if (rank == 0 && !s_got_work && stream.count() == 0) {
                        atomicAdd((int32_t *) &global.termiante_count, 1);
                    }
                    __gbar.Sync();

                    // if a block oposes termination (i.e. not all blocks second
                    // the decision) then termination is aborted
                    if (idx == 0) {
                        if (global.termiante_count != num_blocks) {
                            // printf("Block %d noted failed vote\n", blockIdx.x);
                            global.termiante_count = 0;
                            global.terminate_vote = 0;
                        } else {
                            // printf("Block %d accepts termination\n", blockIdx.x);
                            global.terminate = 1;
                        }
                    }
                    __gbar.Sync();
                }

                // process working_item TODO support dynamic tb maps
                if (s_got_work) {
                    // if (rank == 0) printf("block %d enters working life\n", blockIdx.x);
                    if (l_got_work) {
                        contents(idx, work_item);
                        // __threadfence();
                        
                        // printf("Thread %d of block %d sees Element 0 in stream as: v = %u, d = %u\n", rank, blockIdx.x, stream.read(0).vertex, stream.read(0).depth);
                    }
                    block.sync();
                    if (rank == 0) stream.commit_pending();
                }
            }
            if (rank == 0) printf("Block %d terminating\n", blockIdx.x);
            assert(stream.count() == 0);
            // assert(*stream.m_pending == *stream.m_end);
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
