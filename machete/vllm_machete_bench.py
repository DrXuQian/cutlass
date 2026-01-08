#!/usr/bin/env python3
"""
vLLM Machete Benchmark for NCU Profiling

This script runs vLLM's Machete kernel once for NCU profiling comparison.
Use with: ncu --set full python vllm_machete_bench.py --m=1 --n=2048 --k=2048 --g=128

Compare results with: ncu --set full ./machete_ncu_bench --m=1 --n=2048 --k=2048 --g=128
"""

import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='vLLM Machete NCU Benchmark')
    parser.add_argument('--m', type=int, default=1, help='M dimension (batch/sequence)')
    parser.add_argument('--n', type=int, default=2048, help='N dimension (output features)')
    parser.add_argument('--k', type=int, default=2048, help='K dimension (input features)')
    parser.add_argument('--g', type=int, default=128, help='Group size for quantization')
    args = parser.parse_args()

    M, N, K, G = args.m, args.n, args.k, args.g

    print(f"=== vLLM Machete NCU Benchmark ===")
    print(f"Problem: M={M}, N={N}, K={K}, G={G}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Try to import vLLM's machete ops
    try:
        from vllm._custom_ops import machete_gemm, machete_prepack_B
        print("Using vLLM machete ops")
        use_vllm = True
    except ImportError:
        print("vLLM machete ops not available, using torch.mm fallback")
        use_vllm = False

    # Prepare tensors
    # A: activation matrix (M, K) in BF16
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    if use_vllm:
        # B: weight matrix (N, K) in INT4 (packed as uint8, 2 values per byte)
        # For vLLM machete, weights need to be prepacked
        B_fp = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        # Quantize to INT4
        # Simple symmetric quantization for benchmark
        scale_k = (K + G - 1) // G
        scales = torch.ones(N, scale_k, dtype=torch.bfloat16, device=device)

        # Create fake INT4 weights (packed as int32 for machete)
        B_int4 = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)

        # Prepack weights for machete
        try:
            B_packed = machete_prepack_B(B_int4, scales)
            print("Weights prepacked successfully")
        except Exception as e:
            print(f"Prepack failed: {e}")
            use_vllm = False

    if use_vllm:
        # Warmup
        torch.cuda.synchronize()

        print("Running single kernel launch for NCU profiling...")

        # Single kernel launch for NCU profiling
        try:
            output = machete_gemm(A, B_packed, scales, group_size=G)
            torch.cuda.synchronize()
            print(f"Kernel completed. Output shape: {output.shape}")
        except Exception as e:
            print(f"Machete GEMM failed: {e}")
            use_vllm = False

    if not use_vllm:
        # Fallback: use torch matmul for demonstration
        print("Using torch.mm fallback...")

        B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()

        print("Running single kernel launch for NCU profiling...")
        output = torch.mm(A, B)
        torch.cuda.synchronize()

        print(f"Kernel completed. Output shape: {output.shape}")

    print("Done.")

if __name__ == "__main__":
    main()
