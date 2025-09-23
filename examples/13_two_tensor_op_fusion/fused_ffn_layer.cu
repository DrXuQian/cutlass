/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This example demonstrates fusing a complete FFN (Feed-Forward Network) layer commonly used in
 * transformer models like LLaMA/GPT. The fusion includes:
 *
 * 本示例演示了在LLaMA/GPT等transformer模型中常用的完整FFN（前馈网络）层的融合。
 * 融合包括以下步骤：
 *
 * Input: [9600, 1024]  // 输入: [序列长度*批量大小, 隐藏维度]
 * 1. GEMM1: [9600, 1024] x [1024, 2730] -> [9600, 2730] (gate projection)  // 门控投影
 * 2. GEMM2: [9600, 1024] x [1024, 2730] -> [9600, 2730] (up projection)    // 上投影
 * 3. SiLU activation on GEMM1 output  // 对GEMM1输出应用SiLU激活函数
 * 4. Element-wise multiplication: SiLU(GEMM1) * GEMM2  // 逐元素乘法：SiLU(GEMM1) * GEMM2
 * 5. LayerNorm on the multiplication result  // 对乘法结果进行层归一化
 * 6. GEMM3: [9600, 2730] x [2730, 1024] -> [9600, 1024] (down projection)  // 下投影
 *
 * This mimics the MLP layer in modern transformer architectures.
 * 这模拟了现代transformer架构中的MLP层。
 **************************************************************************************************/

#include <iostream>       // C++标准输入输出流
#include <vector>         // C++标准向量容器
#include <cmath>          // C数学库（用于exp, sqrt等函数）
#include <cuda_runtime.h> // CUDA运行时API
#include <cublas_v2.h>    // cuBLAS库（虽然这里未使用，但可以用于对比）

#include "cutlass/cutlass.h"                               // CUTLASS核心头文件
#include "cutlass/gemm/device/gemm.h"                      // CUTLASS设备端GEMM操作
#include "cutlass/util/host_tensor.h"                      // CUTLASS主机端张量容器
#include "cutlass/util/tensor_view_io.h"                   // 张量视图输入/输出工具
#include "cutlass/util/reference/host/tensor_fill.h"       // 张量填充工具（用于初始化）
#include "cutlass/util/reference/host/tensor_copy.h"       // 张量复制工具
#include "cutlass/util/reference/host/tensor_compare.h"    // 张量比较工具（用于验证）
#include "cutlass/util/reference/host/gemm.h"              // 参考实现的GEMM（用于验证）

////////////////////////////////////////////////////////////////////////////////

// Problem sizes for FFN layer  // FFN层的问题规模
// Typical LLaMA-style dimensions  // 典型的LLaMA风格维度
constexpr int kSeqLength = 9600;   // Sequence length * batch size  // 序列长度 * 批量大小
constexpr int kHiddenDim = 1024;   // Model hidden dimension  // 模型隐藏层维度
constexpr int kFFNDim = 2730;      // FFN intermediate dimension (typically 8/3 * hidden_dim)  // FFN中间层维度（通常是隐藏维度的8/3）

////////////////////////////////////////////////////////////////////////////////

// Simple SiLU activation kernel  // 简单的SiLU激活函数内核
__global__ void silu_multiply_kernel(
    cutlass::half_t const* gate,    // 门控投影的输出（GEMM1的结果）
    cutlass::half_t const* up,      // 上投影的输出（GEMM2的结果）
    cutlass::half_t* output,         // 输出张量（存储SiLU(gate) * up的结果）
    int size                         // 总元素数量
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算全局线程索引
    if (idx < size) {  // 边界检查
        float gate_val = float(gate[idx]);  // 将half转换为float进行计算
        float up_val = float(up[idx]);      // 将half转换为float进行计算

        // SiLU(x) = x * sigmoid(x)  // SiLU激活函数的定义：x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-gate_val));  // 计算sigmoid函数
        float silu = gate_val * sigmoid;                  // 计算SiLU激活值

        output[idx] = cutlass::half_t(silu * up_val);     // 将结果转换回half并存储
    }
}

// Simple LayerNorm kernel  // 简单的层归一化内核
__global__ void layernorm_kernel(
    cutlass::half_t const* input,    // 输入张量
    cutlass::half_t* output,          // 输出张量（归一化后的结果）
    int seq_length,                   // 序列长度（行数）
    int hidden_dim,                   // 隐藏维度（列数）
    float eps = 1e-5f                 // 数值稳定性的小量（避免除零）
) {
    int row = blockIdx.x;  // 每个block处理一行
    if (row < seq_length) {  // 边界检查
        // Compute mean  // 计算均值
        float mean = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {  // 遍历该行的所有元素
            mean += float(input[row * hidden_dim + i]);  // 累加元素值
        }
        mean /= hidden_dim;  // 计算平均值

        // Compute variance  // 计算方差
        float variance = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {  // 再次遍历该行
            float diff = float(input[row * hidden_dim + i]) - mean;  // 计算与均值的差
            variance += diff * diff;  // 累加平方差
        }
        variance /= hidden_dim;  // 计算方差

        // Normalize  // 归一化
        float stddev = sqrtf(variance + eps);  // 计算标准差（加eps避免除零）
        for (int i = 0; i < hidden_dim; ++i) {  // 第三次遍历该行
            output[row * hidden_dim + i] = cutlass::half_t(
                (float(input[row * hidden_dim + i]) - mean) / stddev  // 标准化：(x - mean) / stddev
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {  // 主函数入口

    std::cout << "=== Fused FFN Layer Example ===\n";  // 融合FFN层示例标题
    std::cout << "This example demonstrates fusing a complete transformer FFN layer:\n";  // 说明这是一个完整的transformer FFN层融合示例
    std::cout << "Input[" << kSeqLength << "," << kHiddenDim << "] -> ";  // 输入维度
    std::cout << "GEMM -> [" << kSeqLength << "," << kFFNDim << "] -> ";  // 第一阶段GEMM后的维度
    std::cout << "SiLU*Up -> LayerNorm -> GEMM -> [" << kSeqLength << "," << kHiddenDim << "]\n\n";  // 完整的处理流程

    // Check GPU  // 检查GPU设备
    cudaDeviceProp props;  // CUDA设备属性结构体
    cudaError_t error = cudaGetDeviceProperties(&props, 0);  // 获取设备0的属性
    if (error != cudaSuccess) {  // 如果获取失败
        std::cerr << "cudaGetDeviceProperties() failed: " << cudaGetErrorString(error) << "\n";  // 输出错误信息
        return -1;  // 返回错误代码
    }

    std::cout << "Running on GPU: " << props.name << " (SM" << props.major << props.minor << ")\n";  // 输出GPU名称和计算能力

    // Define data types  // 定义数据类型
    using ElementInput = cutlass::half_t;      // 输入元素类型：半精度浮点数（FP16）
    using ElementOutput = cutlass::half_t;     // 输出元素类型：半精度浮点数（FP16）
    using ElementAccumulator = float;          // 累加器元素类型：单精度浮点数（FP32）

    using LayoutInput = cutlass::layout::RowMajor;     // 输入布局：行主序（每行连续存储）
    using LayoutWeight = cutlass::layout::ColumnMajor;  // 权重布局：列主序（每列连续存储）
    using LayoutOutput = cutlass::layout::RowMajor;    // 输出布局：行主序（每行连续存储）

    // Allocate host tensors  // 分配主机端张量
    cutlass::HostTensor<ElementInput, LayoutInput> tensor_input({kSeqLength, kHiddenDim});          // 输入张量 [9600, 1024]
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_gate_weight({kHiddenDim, kFFNDim});      // 门控权重 [1024, 2730]
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_up_weight({kHiddenDim, kFFNDim});        // 上投影权重 [1024, 2730]
    cutlass::HostTensor<ElementInput, LayoutWeight> tensor_down_weight({kFFNDim, kHiddenDim});      // 下投影权重 [2730, 1024]
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_output({kSeqLength, kHiddenDim});       // 最终输出 [9600, 1024]

    // Intermediate tensors  // 中间张量
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_gate_out({kSeqLength, kFFNDim});    // 门控投影输出 [9600, 2730]
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_up_out({kSeqLength, kFFNDim});      // 上投影输出 [9600, 2730]
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_activated({kSeqLength, kFFNDim});   // SiLU激活后的输出 [9600, 2730]
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_normed({kSeqLength, kFFNDim});      // 层归一化后的输出 [9600, 2730]

    // Initialize input tensors with random data  // 用随机数据初始化输入张量
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_input.host_view(),  // 要填充的张量视图
        1,                          // 随机数生成的种子
        ElementInput(2),            // 随机数范围的最大值
        ElementInput(-2),           // 随机数范围的最小值
        0                           // 偏移（通常为0）
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_gate_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        1
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_up_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        2
    );

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_down_weight.host_view(),
        1,
        ElementInput(0.5),
        ElementInput(-0.5),
        3
    );

    // Copy to device  // 复制到设备端（GPU）
    tensor_input.sync_device();         // 同步输入张量到GPU
    tensor_gate_weight.sync_device();   // 同步门控权重到GPU
    tensor_up_weight.sync_device();     // 同步上投影权重到GPU
    tensor_down_weight.sync_device();   // 同步下投影权重到GPU

    std::cout << "Executing FFN operations...\n";

    // Define GEMM operation for FP16  // 定义FP16的GEMM操作
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, LayoutInput,           // A matrix  // A矩阵：输入类型和布局
        ElementInput, LayoutWeight,           // B matrix  // B矩阵：权重类型和布局
        ElementOutput, LayoutOutput,          // C matrix  // C矩阵：输出类型和布局
        ElementAccumulator                    // Accumulator  // 累加器类型（FP32用于更高精度）
    >;

    // GEMM1: Gate projection  // GEMM1：门控投影
    Gemm gemm_gate;  // 创建门控GEMM操作对象
    typename Gemm::Arguments args_gate(  // 配置GEMM参数
        {kSeqLength, kFFNDim, kHiddenDim},   // Problem size  // 问题规模：M=9600, N=2730, K=1024
        tensor_input.device_ref(),            // A  // A矩阵：输入张量 [9600, 1024]
        tensor_gate_weight.device_ref(),      // B  // B矩阵：门控权重 [1024, 2730]
        tensor_gate_out.device_ref(),         // C (unused)  // C矩阵（未使用，因为beta=0）
        tensor_gate_out.device_ref(),         // D (output)  // D矩阵：输出 [9600, 2730]
        {ElementAccumulator(1), ElementAccumulator(0)}  // alpha, beta  // D = alpha*A*B + beta*C，这里alpha=1, beta=0
    );

    cutlass::Status status = gemm_gate(args_gate);  // 执行门控GEMM操作
    if (status != cutlass::Status::kSuccess) {  // 检查执行状态
        std::cerr << "Gate GEMM failed\n";  // 如果失败，输出错误信息
        return -1;  // 返回错误代码
    }

    // GEMM2: Up projection  // GEMM2：上投影
    Gemm gemm_up;  // 创建上投影GEMM操作对象
    typename Gemm::Arguments args_up(  // 配置GEMM参数
        {kSeqLength, kFFNDim, kHiddenDim},  // 问题规模：与门控投影相同
        tensor_input.device_ref(),           // A矩阵：同样使用输入张量 [9600, 1024]
        tensor_up_weight.device_ref(),       // B矩阵：上投影权重 [1024, 2730]
        tensor_up_out.device_ref(),          // C矩阵（未使用）
        tensor_up_out.device_ref(),          // D矩阵：输出 [9600, 2730]
        {ElementAccumulator(1), ElementAccumulator(0)}  // alpha=1, beta=0
    );

    status = gemm_up(args_up);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Up GEMM failed\n";
        return -1;
    }

    // Apply SiLU activation and multiply  // 应用SiLU激活函数并进行逐元素乘法
    dim3 block(256);  // 每个block有256个线程
    dim3 grid((kSeqLength * kFFNDim + block.x - 1) / block.x);  // 计算需要的grid大小（向上取整）
    silu_multiply_kernel<<<grid, block>>>(  // 启动CUDA内核
        tensor_gate_out.device_data(),      // 门控投影的输出（将应用SiLU）
        tensor_up_out.device_data(),        // 上投影的输出（将与SiLU结果相乘）
        tensor_activated.device_data(),     // 输出：SiLU(gate) * up
        kSeqLength * kFFNDim                // 总元素数：9600 * 2730
    );

    // Apply LayerNorm  // 应用层归一化
    layernorm_kernel<<<kSeqLength, 1>>>(  // 每行使用一个block（简单实现，非最优）
        tensor_activated.device_data(),   // 输入：激活后的张量
        tensor_normed.device_data(),      // 输出：归一化后的张量
        kSeqLength,                       // 序列长度（行数）
        kFFNDim                           // FFN维度（列数）
    );

    // GEMM3: Down projection  // GEMM3：下投影
    Gemm gemm_down;  // 创建下投影GEMM操作对象
    typename Gemm::Arguments args_down(  // 配置GEMM参数
        {kSeqLength, kHiddenDim, kFFNDim},  // 问题规模：M=9600, N=1024, K=2730
        tensor_normed.device_ref(),          // A矩阵：归一化后的张量 [9600, 2730]
        tensor_down_weight.device_ref(),     // B矩阵：下投影权重 [2730, 1024]
        tensor_output.device_ref(),          // C矩阵（未使用）
        tensor_output.device_ref(),          // D矩阵：最终输出 [9600, 1024]
        {ElementAccumulator(1), ElementAccumulator(0)}  // alpha=1, beta=0
    );

    status = gemm_down(args_down);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Down GEMM failed\n";
        return -1;
    }

    // Synchronize and copy back  // 同步并复制回主机
    cudaDeviceSynchronize();  // 等待所有CUDA操作完成
    tensor_output.sync_host();  // 将输出张量从GPU复制回CPU

    std::cout << "FFN layer execution completed successfully!\n\n";

    // Performance measurement  // 性能测量
    std::cout << "=== Performance Benchmark ===\n";  // 性能基准测试标题
    std::cout << "Problem size: [" << kSeqLength << ", " << kHiddenDim << "] -> ["  // 输出问题规模
              << kSeqLength << ", " << kFFNDim << "] -> [" << kSeqLength << ", " << kHiddenDim << "]\n";  // 显示完整的数据流维度变化

    cudaEvent_t start, stop;  // CUDA事件，用于精确计时
    cudaEventCreate(&start);   // 创建开始事件
    cudaEventCreate(&stop);    // 创建结束事件

    const int num_iterations = 100;  // 迭代次数，用于取平均值

    // Warm-up  // 预热（让GPU达到稳定状态）
    for (int i = 0; i < 10; ++i) {  // 执行10次预热迭代
        gemm_gate(args_gate);
        gemm_up(args_up);
        silu_multiply_kernel<<<grid, block>>>(
            tensor_gate_out.device_data(),
            tensor_up_out.device_data(),
            tensor_activated.device_data(),
            kSeqLength * kFFNDim
        );
        layernorm_kernel<<<kSeqLength, 1>>>(
            tensor_activated.device_data(),
            tensor_normed.device_data(),
            kSeqLength,
            kFFNDim
        );
        gemm_down(args_down);
    }
    cudaDeviceSynchronize();

    // Benchmark  // 性能基准测试
    cudaEventRecord(start);  // 记录开始时间
    for (int i = 0; i < num_iterations; ++i) {  // 执行100次迭代
        gemm_gate(args_gate);
        gemm_up(args_up);
        silu_multiply_kernel<<<grid, block>>>(
            tensor_gate_out.device_data(),
            tensor_up_out.device_data(),
            tensor_activated.device_data(),
            kSeqLength * kFFNDim
        );
        layernorm_kernel<<<kSeqLength, 1>>>(
            tensor_activated.device_data(),
            tensor_normed.device_data(),
            kSeqLength,
            kFFNDim
        );
        gemm_down(args_down);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;  // 存储经过的时间（毫秒）
    cudaEventElapsedTime(&time_ms, start, stop);  // 计算start和stop之间的时间差

    std::cout << "Average time per FFN layer: " << time_ms / num_iterations << " ms\n";  // 输出每个FFN层的平均执行时间

    // Calculate FLOPs  // 计算浮点运算次数
    double flops = 2.0 * kSeqLength * kHiddenDim * kFFNDim * 2 +  // Two GEMMs to FFN dim  // 两个GEMM到FFN维度（门控和上投影）
                   2.0 * kSeqLength * kFFNDim * kHiddenDim +       // Down projection  // 下投影GEMM
                   kSeqLength * kFFNDim * 5;                        // SiLU and multiply  // SiLU激活和乘法运算

    double tflops = (flops * num_iterations) / (time_ms * 1e9);  // 计算TFLOPS（每秒万亿次浮点运算）
    std::cout << "Performance: " << tflops << " TFLOPS\n";  // 输出性能指标

    // Memory bandwidth  // 内存带宽计算
    double bytes = sizeof(ElementInput) * (  // 计算总字节数
        kSeqLength * kHiddenDim +           // Input  // 输入数据
        kHiddenDim * kFFNDim * 2 +           // Gate and up weights  // 门控和上投影权重
        kFFNDim * kHiddenDim +               // Down weight  // 下投影权重
        kSeqLength * kFFNDim * 4 +           // Intermediate results  // 中间结果（gate_out, up_out, activated, normed）
        kSeqLength * kHiddenDim              // Output  // 输出数据
    );
    double bandwidth = (bytes * num_iterations) / (time_ms * 1e6);  // 计算带宽（GB/s）
    std::cout << "Memory bandwidth: " << bandwidth << " GB/s\n";  // 输出内存带宽

    std::cout << "\n=== Fusion Opportunities ===\n";  // 融合优化机会
    std::cout << "1. Fuse gate and up GEMMs (share input loading)\n";  // 1. 融合门控和上投影GEMM（共享输入加载）
    std::cout << "2. Fuse SiLU activation with GEMM epilogue\n";  // 2. 将SiLU激活函数融合到GEMM的epilogue阶段
    std::cout << "3. Keep intermediate results in shared memory\n";  // 3. 将中间结果保存在共享内存中
    std::cout << "4. Fuse LayerNorm with down projection prologue\n";  // 4. 将层归一化融合到下投影的prologue阶段
    std::cout << "5. Use persistent kernels to avoid global memory traffic\n";  // 5. 使用持久化内核避免全局内存访问

    cudaEventDestroy(start);  // 销毁开始事件
    cudaEventDestroy(stop);   // 销毁结束事件

    return 0;  // 程序正常退出
}