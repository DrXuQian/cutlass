#include <cstdio>
#include <cuda_runtime.h>

// elect_one_sync implementation (from CuTe)
__device__ __forceinline__
int elect_one_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  elect.sync _|p, 0xFFFFFFFF;\n"
        "  selp.b32 %0, 1, 0, p;\n"
        "  mov.u32 %1, %%laneid;\n"
        "}\n"
        : "=r"(pred), "=r"(laneid)
    );
    return pred;
#else
    // Fallback for older architectures
    uint32_t mask = __activemask();
    uint32_t leader = __ffs(mask) - 1;
    return (threadIdx.x % 32) == leader;
#endif
}

// Kernel demonstrating the difference
__global__ void demo_elect_one(int* results, bool use_divergent_branch) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Simulate divergent execution: only odd lanes active
    if (use_divergent_branch) {
        if (lane_id % 2 == 0) {
            // Even lanes exit early - Lane 0 is NOT active!
            return;
        }
    }

    // Method 1: Simple check (WRONG if Lane 0 not active!)
    bool leader_simple = (lane_id == 0);

    // Method 2: elect_one_sync (CORRECT - picks first active lane)
    bool leader_elect = elect_one_sync();

    // Store results
    if (leader_simple) {
        atomicAdd(&results[warp_id * 2], 1);  // Simple method count
    }
    if (leader_elect) {
        atomicAdd(&results[warp_id * 2 + 1], 1);  // Elect method count
    }
}

// Kernel to show which lane becomes leader
__global__ void show_leader_lane(int* leader_lanes, bool divergent) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Simulate divergence: only lanes 5-31 active
    if (divergent && lane_id < 5) {
        return;
    }

    bool is_leader = elect_one_sync();
    if (is_leader) {
        leader_lanes[warp_id] = lane_id;
    }
}

int main() {
    printf("============================================================\n");
    printf("    0x02 elect_one_sync vs Simple Leader Election\n");
    printf("============================================================\n\n");

    //==========================================================================
    // Test 1: Normal execution (all lanes active)
    //==========================================================================

    printf("=== Test 1: All Lanes Active ===\n");
    {
        int* d_results;
        cudaMalloc(&d_results, 16 * sizeof(int));
        cudaMemset(d_results, 0, 16 * sizeof(int));

        demo_elect_one<<<1, 256>>>(d_results, false);
        cudaDeviceSynchronize();

        int h_results[16];
        cudaMemcpy(h_results, d_results, 16 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("8 warps, all lanes active:\n");
        printf("Warp | Simple | Elect\n");
        printf("-----|--------|------\n");
        for (int w = 0; w < 8; ++w) {
            printf("  %d  |   %d    |   %d\n", w, h_results[w*2], h_results[w*2+1]);
        }
        printf("\nBoth methods work correctly when all lanes are active.\n");

        cudaFree(d_results);
    }

    //==========================================================================
    // Test 2: Divergent execution (only odd lanes active)
    //==========================================================================

    printf("\n=== Test 2: Divergent Execution (Only Odd Lanes Active) ===\n");
    {
        int* d_results;
        cudaMalloc(&d_results, 16 * sizeof(int));
        cudaMemset(d_results, 0, 16 * sizeof(int));

        demo_elect_one<<<1, 256>>>(d_results, true);
        cudaDeviceSynchronize();

        int h_results[16];
        cudaMemcpy(h_results, d_results, 16 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("8 warps, only ODD lanes active (Lane 0 is INACTIVE!):\n");
        printf("Warp | Simple | Elect\n");
        printf("-----|--------|------\n");
        for (int w = 0; w < 8; ++w) {
            printf("  %d  |   %d    |   %d\n", w, h_results[w*2], h_results[w*2+1]);
        }
        printf("\nSimple method: Lane 0 is inactive, so NO leader elected! (0)\n");
        printf("Elect method:  First active lane (Lane 1) becomes leader. (1)\n");

        cudaFree(d_results);
    }

    //==========================================================================
    // Test 3: Show which lane becomes leader
    //==========================================================================

    printf("\n=== Test 3: Which Lane Becomes Leader ===\n");
    {
        int* d_leaders;
        cudaMalloc(&d_leaders, 8 * sizeof(int));

        // Normal case
        cudaMemset(d_leaders, -1, 8 * sizeof(int));
        show_leader_lane<<<1, 256>>>(d_leaders, false);
        cudaDeviceSynchronize();

        int h_leaders[8];
        cudaMemcpy(h_leaders, d_leaders, 8 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Normal (all active): Leader lanes = ");
        for (int w = 0; w < 8; ++w) printf("%d ", h_leaders[w]);
        printf("\n");

        // Divergent case: lanes 0-4 inactive
        cudaMemset(d_leaders, -1, 8 * sizeof(int));
        show_leader_lane<<<1, 256>>>(d_leaders, true);
        cudaDeviceSynchronize();

        cudaMemcpy(h_leaders, d_leaders, 8 * sizeof(int), cudaMemcpyDeviceToHost);

        printf("Divergent (lanes 0-4 inactive): Leader lanes = ");
        for (int w = 0; w < 8; ++w) printf("%d ", h_leaders[w]);
        printf("\n");

        printf("\nWith divergence, Lane 5 becomes the leader (first active lane).\n");

        cudaFree(d_leaders);
    }

    printf("\n============================================================\n");
    printf("Summary:\n");
    printf("  - Simple (lane == 0): FAILS if Lane 0 is inactive\n");
    printf("  - elect_one_sync():   Always picks first ACTIVE lane\n");
    printf("  - Use elect_one_sync() for robustness in divergent code\n");
    printf("============================================================\n");

    return 0;
}
