/*
 * Standalone TRT-LLM MoE Grouped GEMM Benchmark
 *
 * This benchmark uses the extracted TRT-LLM kernel for Hopper (SM90).
 *
 * Usage:
 *   ./moe_gemm_trtllm_bench [--num_experts=8] [--tokens=128] [--hidden=4096] [--inter=11008]
 */

#include "moe_gemm_trtllm.cuh"
#include <chrono>
#include <random>

struct Options {
    int num_experts = 8;
    int tokens_per_expert = 128;
    int hidden_size = 4096;
    int inter_size = 11008;
    int warmup = 10;
    int iterations = 100;
    bool help = false;

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.find("--num_experts=") == 0) num_experts = std::stoi(arg.substr(14));
            else if (arg.find("--tokens=") == 0) tokens_per_expert = std::stoi(arg.substr(9));
            else if (arg.find("--hidden=") == 0) hidden_size = std::stoi(arg.substr(9));
            else if (arg.find("--inter=") == 0) inter_size = std::stoi(arg.substr(8));
            else if (arg.find("--warmup=") == 0) warmup = std::stoi(arg.substr(9));
            else if (arg.find("--iterations=") == 0) iterations = std::stoi(arg.substr(13));
            else if (arg == "--help" || arg == "-h") help = true;
        }
    }

    void print_usage() {
        std::cout << "TRT-LLM MoE Grouped GEMM Benchmark\n"
                  << "Options:\n"
                  << "  --num_experts=N    Number of experts (default: 8)\n"
                  << "  --tokens=N         Tokens per expert (default: 128)\n"
                  << "  --hidden=N         Hidden size (default: 4096)\n"
                  << "  --inter=N          Intermediate size (default: 11008)\n"
                  << "  --warmup=N         Warmup iterations (default: 10)\n"
                  << "  --iterations=N     Benchmark iterations (default: 100)\n";
    }
};

// Initialize with random values
void init_random(half* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<half> host_data(size);
    for (size_t i = 0; i < size; i++) {
        host_data[i] = __float2half(dist(gen));
    }
    CUDA_CHECK(cudaMemcpy(data, host_data.data(), size * sizeof(half), cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    Options opts;
    opts.parse(argc, argv);

    if (opts.help) {
        opts.print_usage();
        return 0;
    }

    std::cout << "=== TRT-LLM MoE Grouped GEMM Benchmark (CUTLASS 3.x) ===" << std::endl;

    // Check CUDA device
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "SM: " << props.major << "." << props.minor << std::endl;
    std::cout << "SM Count: " << props.multiProcessorCount << std::endl;

    if (props.major < 9) {
        std::cout << "This test requires Hopper (SM90) or newer" << std::endl;
        return 0;
    }

    int M = opts.tokens_per_expert;
    int N = opts.inter_size;
    int K = opts.hidden_size;
    int num_experts = opts.num_experts;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Experts: " << num_experts << std::endl;
    std::cout << "  Tokens/expert (M): " << M << std::endl;
    std::cout << "  Intermediate (N): " << N << std::endl;
    std::cout << "  Hidden (K): " << K << std::endl;

    // Memory sizes per expert
    size_t size_A_per_expert = (size_t)M * K;
    size_t size_B_per_expert = (size_t)K * N;
    size_t size_C_per_expert = (size_t)M * N;

    std::cout << "\nMemory per expert:" << std::endl;
    std::cout << "  A: " << size_A_per_expert * sizeof(half) / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  B: " << size_B_per_expert * sizeof(half) / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  C: " << size_C_per_expert * sizeof(half) / 1024.0 / 1024.0 << " MB" << std::endl;

    // Allocate device memory for each expert
    std::vector<half*> d_A(num_experts);
    std::vector<half*> d_B(num_experts);
    std::vector<half*> d_C(num_experts);

    for (int i = 0; i < num_experts; i++) {
        CUDA_CHECK(cudaMalloc(&d_A[i], size_A_per_expert * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B[i], size_B_per_expert * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C[i], size_C_per_expert * sizeof(half)));

        // Initialize with random values
        init_random(d_A[i], size_A_per_expert);
        init_random(d_B[i], size_B_per_expert);
        CUDA_CHECK(cudaMemset(d_C[i], 0, size_C_per_expert * sizeof(half)));
    }

    std::cout << "\nMemory allocated for " << num_experts << " experts" << std::endl;

    // Allocate pointer arrays on device
    half const** d_ptr_A;
    half const** d_ptr_B;
    half** d_ptr_C;

    CUDA_CHECK(cudaMalloc(&d_ptr_A, num_experts * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_ptr_B, num_experts * sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_ptr_C, num_experts * sizeof(half*)));

    CUDA_CHECK(cudaMemcpy(d_ptr_A, d_A.data(), num_experts * sizeof(half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr_B, d_B.data(), num_experts * sizeof(half*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr_C, d_C.data(), num_experts * sizeof(half*), cudaMemcpyHostToDevice));

    // Create runner
    moe_gemm::SimpleMoeGemmRunner<half> runner;

    // Get and allocate workspace
    size_t workspace_size = runner.get_workspace_size(num_experts);
    std::cout << "Workspace size: " << workspace_size / 1024.0 << " KB" << std::endl;

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    std::cout << "\nRunning " << opts.warmup << " warmup iterations..." << std::endl;
    for (int i = 0; i < opts.warmup; i++) {
        runner.run(num_experts, M, N, K, d_ptr_A, d_ptr_B, d_ptr_C, workspace, workspace_size, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    std::cout << "Running " << opts.iterations << " benchmark iterations..." << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < opts.iterations; i++) {
        runner.run(num_experts, M, N, K, d_ptr_A, d_ptr_B, d_ptr_C, workspace, workspace_size, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Calculate metrics
    float avg_time_ms = elapsed_ms / opts.iterations;
    double total_flops = 2.0 * M * N * K * num_experts;  // 2 * M * N * K per expert
    double tflops = total_flops / (avg_time_ms * 1e9);   // TFLOPS

    // Memory bandwidth (rough estimate)
    double bytes_read = (double)(size_A_per_expert + size_B_per_expert) * sizeof(half) * num_experts;
    double bytes_written = (double)size_C_per_expert * sizeof(half) * num_experts;
    double total_bytes = bytes_read + bytes_written;
    double bandwidth_tb_s = total_bytes / (avg_time_ms * 1e9);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;
    std::cout << "Memory bandwidth: " << bandwidth_tb_s << " TB/s" << std::endl;
    std::cout << "Throughput: " << 1000.0 / avg_time_ms << " calls/sec" << std::endl;

    // Theoretical peak comparison for H100
    double peak_tflops_fp16 = 989.4;  // H100 SXM FP16 Tensor Core
    std::cout << "Efficiency vs H100 peak: " << (tflops / peak_tflops_fp16 * 100.0) << "%" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }

    CUDA_CHECK(cudaFree(d_ptr_A));
    CUDA_CHECK(cudaFree(d_ptr_B));
    CUDA_CHECK(cudaFree(d_ptr_C));

    for (int i = 0; i < num_experts; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    return 0;
}
