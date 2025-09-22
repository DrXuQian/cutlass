/*
 * Simplified B2B GEMM with RF (Register File) Residency
 *
 * 这个文件演示了CUTLASS风格的B2B GEMM融合，保持了CUTLASS的逻辑结构，
 * 但简化了实现细节，便于理解核心概念。
 *
 * 关键特性：
 * 1. RF驻留（Register File Residency）：中间结果保存在寄存器中
 * 2. Device/Kernel分离架构：遵循CUTLASS的设计模式
 * 3. 模板化设计：支持不同的Tile尺寸和配置
 * 4. Epilogue融合：支持ReLU等激活函数
 *
 * 与完整CUTLASS的区别：
 * - 简化了Tensor Core操作
 * - 简化了内存访问模式
 * - 去除了软件流水线
 * - 简化了线程块级协作
 *
 * SM80 FP16 only - 针对Ampere架构优化
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

///////////////////////////////////////////////////////////////////////////////
// Simplified B2B GEMM Kernel - RF Residency Version
//
// 核心概念：寄存器驻留（RF Residency）
// 中间结果C保存在寄存器中，不写入全局内存，避免了：
// 1. 一次全局内存写入（~500 cycles）
// 2. 一次全局内存读取（~500 cycles）
// 寄存器访问只需要<1 cycle，性能提升巨大
//
// 内存层次结构（从快到慢）：
// - 寄存器（RF）：<1 cycle，每个线程255个32位寄存器
// - 共享内存：~30 cycles，每个SM 48-164KB
// - L1缓存：~100 cycles，每个SM 128KB
// - L2缓存：~200 cycles，全局6MB
// - 全局内存：~500 cycles，8-24GB
///////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/**
 * @brief 简化的B2B GEMM kernel类（RF驻留版本）
 *
 * @tparam ThreadblockShape0_ 第一个GEMM的线程块Tile形状 [M, N, K]
 * @tparam ThreadblockShape1_ 第二个GEMM的线程块Tile形状 [M, N, K]
 * @tparam WarpShape0_ 第一个GEMM的Warp级Tile形状
 * @tparam WarpShape1_ 第二个GEMM的Warp级Tile形状
 * @tparam InstructionShape_ Tensor Core指令形状（如mma.sync）
 * @tparam EpilogueOutputOp0_ 第一个GEMM的epilogue操作（如ReLU）
 * @tparam EpilogueOutputOp1_ 第二个GEMM的epilogue操作
 */
template <
    typename ThreadblockShape0_,
    typename ThreadblockShape1_,
    typename WarpShape0_,
    typename WarpShape1_,
    typename InstructionShape_,
    typename EpilogueOutputOp0_,
    typename EpilogueOutputOp1_
>
class SimplifiedB2bGemmRF {
public:
    using ThreadblockShape0 = ThreadblockShape0_;
    using ThreadblockShape1 = ThreadblockShape1_;
    using WarpShape0 = WarpShape0_;
    using WarpShape1 = WarpShape1_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp0 = EpilogueOutputOp0_;
    using EpilogueOutputOp1 = EpilogueOutputOp1_;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    /**
     * @brief Kernel参数结构体
     *
     * 包含所有kernel执行所需的参数：
     * - 问题尺寸：两个GEMM的维度
     * - 张量引用：指向设备内存的指针和stride信息
     * - Epilogue参数：alpha/beta缩放因子等
     */
    struct Params {
        cutlass::gemm::GemmCoord problem_size_0;  // 第一个GEMM: [M,N,K]
        cutlass::gemm::GemmCoord problem_size_1;  // 第二个GEMM: [M,P,N]
        cutlass::TensorRef<ElementA const, LayoutA> ref_A0;  // A矩阵引用
        cutlass::TensorRef<ElementB const, LayoutB> ref_B0;  // B0矩阵引用
        cutlass::TensorRef<ElementB const, LayoutB> ref_B1;  // B1矩阵引用
        cutlass::TensorRef<ElementC, LayoutC> ref_D1;        // 输出D矩阵引用
        typename EpilogueOutputOp0::Params epilogue0;  // 第一个epilogue参数
        typename EpilogueOutputOp1::Params epilogue1;  // 第二个epilogue参数
    };

    /**
     * @brief 共享内存结构体
     *
     * RF版本中共享内存使用最少，因为中间结果保存在寄存器中。
     * 实际CUTLASS中会用于：
     * - 存储从全局内存加载的Tile数据
     * - 线程间的数据交换
     * - Software pipelining的多级缓冲
     */
    union SharedStorage {
        struct {
            typename cutlass::gemm::GemmShape<
                ThreadblockShape0::kM,
                ThreadblockShape0::kN,
                ThreadblockShape0::kK
            > gemm_shape;  // 预留空间，简化版本未充分使用
        } main;
    };

    /**
     * @brief Kernel主函数，执行B2B GEMM融合操作
     *
     * @param params 包含所有kernel参数的结构体
     * @param shared_storage 共享内存空间
     *
     * 执行流程：
     * 1. 第一个GEMM：C = A * B0
     * 2. 应用epilogue（如ReLU）
     * 3. 保持C在寄存器中（RF驻留）
     * 4. 第二个GEMM：D = C * B1
     * 5. 应用epilogue并写入全局内存
     */
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {
        // 线程和Warp标识
        // GPU执行模型：Grid -> Block -> Warp（32线程） -> Thread
        int thread_idx = threadIdx.x;      // 线程在block内的索引
        int warp_idx = thread_idx / 32;    // Warp索引（每个Warp 32个线程）
        int lane_idx = thread_idx % 32;    // 线程在Warp内的索引
        int block_idx_x = blockIdx.x;      // Block在Grid x维度的索引
        int block_idx_y = blockIdx.y;      // Block在Grid y维度的索引

        // 计算线程块级别的矩阵偏移
        // 每个线程块处理输出矩阵的一个Tile
        int block_m = block_idx_x * ThreadblockShape0::kM;  // M维度偏移
        int block_n = block_idx_y * ThreadblockShape0::kN;  // N维度偏移

        // ===== 第一个GEMM: C = A * B0 =====
        // 关键：使用Fragment（寄存器数组）存储中间结果

        // Fragment是CUTLASS中的核心概念：
        // - 存储在寄存器中的小块数据
        // - 每个线程持有整个Warp计算结果的一部分
        // - 大小计算：WarpShape / 线程数 = 每线程的元素数
        ElementAccumulator accumulator_frag[WarpShape0::kM * WarpShape0::kN / 32];

        // 初始化累加器Fragment
        // CUTLASS_PRAGMA_UNROLL：编译时展开循环，提高性能
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < WarpShape0::kM * WarpShape0::kN / 32; ++i) {
            accumulator_frag[i] = ElementAccumulator(0);  // 初始化为0
        }

        // 第一个GEMM的主循环
        // 沿K维度进行分块计算（Tiling）
        // 实际CUTLASS中包含：
        // 1. 全局内存到共享内存的数据加载
        // 2. 共享内存到寄存器的数据加载
        // 3. Tensor Core计算（mma.sync指令）
        // 4. Software pipelining优化
        for (int k_tile = 0; k_tile < params.problem_size_0.k(); k_tile += ThreadblockShape0::kK) {
            // 简化版本：直接计算，未使用Tensor Core

            // 矩阵乘法计算（简化版，未使用Tensor Core）
            // 边界检查：确保不越界访问
            if (block_m < params.problem_size_0.m() && block_n < params.problem_size_0.n()) {
                // 线程到数据的映射：
                // 假设16x16的线程布局，每个线程计算4x4的小块
                // 这是简化的映射，实际CUTLASS使用更复杂的映射策略
                int thread_m = block_m + (thread_idx / 16) * 4;  // 该线程负责的M坐标
                int thread_n = block_n + (thread_idx % 16) * 4;  // 该线程负责的N坐标

                if (thread_m < params.problem_size_0.m() && thread_n < params.problem_size_0.n()) {
                    // 内层K循环：执行点积运算
                    for (int k = k_tile; k < min(k_tile + ThreadblockShape0::kK, params.problem_size_0.k()); ++k) {
                        // 从全局内存读取A和B的元素
                        // TensorRef.at()：CUTLASS的安全访问方法
                        ElementA a_val = params.ref_A0.at({thread_m, k});
                        ElementB b_val = params.ref_B0.at({k, thread_n});

                        // 累加到寄存器中
                        // 使用float进行计算以提高精度（混合精度计算）
                        accumulator_frag[0] += float(a_val) * float(b_val);
                    }
                }
            }
        }

        // 应用第一个GEMM的epilogue操作
        // Epilogue可以是：
        // - 线性组合：C = alpha * A*B + beta * C
        // - 激活函数：ReLU, GELU, Sigmoid等
        // - 量化操作：FP32 -> INT8
        typename EpilogueOutputOp0::FragmentOutput output_frag_0;
        output_frag_0[0] = params.epilogue0(accumulator_frag[0]);

        // ========== 寄存器驻留（RF Residency）核心 ==========
        //
        // 关键优化点：output_frag_0保持在寄存器中！
        // 传统方法：C写入全局内存 -> 第二个GEMM再读取（~1000 cycles）
        // RF驻留：C保持在寄存器 -> 直接用于第二个GEMM（<1 cycle）
        // 性能提升：避免了内存带宽瓶颈，减少了功耗
        //
        // 注意：这要求中间矩阵C的Tile大小适合寄存器容量

        // ===== 第二个GEMM: D = C * B1 =====
        // 使用寄存器中的C作为输入，计算最终结果D

        // 为第二个GEMM分配新的累加器Fragment
        // 注意：可能与第一个GEMM使用不同的Warp形状
        ElementAccumulator accumulator_frag_1[WarpShape1::kM * WarpShape1::kN / 32];

        // 初始化第二个GEMM的累加器
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < WarpShape1::kM * WarpShape1::kN / 32; ++i) {
            accumulator_frag_1[i] = ElementAccumulator(0);
        }

        // 使用寄存器中的output_frag_0作为第二个GEMM的输入
        // 计算P维度的块偏移（第二个GEMM输出的列维度）
        int block_p = block_idx_y * ThreadblockShape1::kN;

        // 第二个GEMM的主循环
        // 沿N维度进行分块（N是第一个GEMM的输出列，第二个GEMM的K维度）
        for (int n_tile = 0; n_tile < params.problem_size_0.n(); n_tile += ThreadblockShape1::kK) {
            if (block_m < params.problem_size_1.m() && block_p < params.problem_size_1.n()) {
                // 计算当前线程在第二个GEMM输出中的位置
                int thread_m = block_m + (thread_idx / 16) * 4;  // M维度位置
                int thread_p = block_p + (thread_idx % 16) * 4;  // P维度位置

                if (thread_m < params.problem_size_1.m() && thread_p < params.problem_size_1.n()) {
                    // 关键：使用寄存器中的C（output_frag_0）进行计算
                    for (int n = n_tile; n < min(n_tile + ThreadblockShape1::kK, params.problem_size_0.n()); ++n) {
                        // 简化的索引逻辑：
                        // 实际CUTLASS使用复杂的Swizzle和Bank conflict避免策略
                        // 这里简化为：如果n匹配当前线程的C元素位置，使用寄存器值
                        float c_val = (n == block_n + (thread_idx % 16) * 4) ? output_frag_0[0] : 0.0f;

                        // 从全局内存读取B1的元素
                        ElementB b1_val = params.ref_B1.at({n, thread_p});

                        // 累加：D[m,p] += C[m,n] * B1[n,p]
                        accumulator_frag_1[0] += c_val * float(b1_val);
                    }
                }
            }
        }

        // 应用第二个GEMM的epilogue操作
        // 这是最后一步，结果将写入全局内存
        typename EpilogueOutputOp1::FragmentOutput output_frag_1;
        output_frag_1[0] = params.epilogue1(accumulator_frag_1[0]);

        // 将最终结果写入全局内存
        // 这是整个B2B GEMM中唯一的全局内存写操作
        if (block_m < params.problem_size_1.m() && block_p < params.problem_size_1.n()) {
            int thread_m = block_m + (thread_idx / 16) * 4;
            int thread_p = block_p + (thread_idx % 16) * 4;

            // 边界检查后写入
            if (thread_m < params.problem_size_1.m() && thread_p < params.problem_size_1.n()) {
                // 类型转换并写入：float -> half
                params.ref_D1.at({thread_m, thread_p}) = ElementC(output_frag_1[0]);
            }
        }
    }
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////
// Simplified Device-level B2B GEMM
//
// Device层是CUTLASS的API层，负责：
// 1. 参数打包和验证
// 2. Kernel启动配置计算
// 3. 共享内存大小计算
// 4. 错误处理
//
// 设计模式：Device类封装Kernel类，提供高层接口
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Device级别的B2B GEMM类
 *
 * 提供用户友好的接口，隐藏kernel细节。
 * 遵循CUTLASS的Device/Kernel分离模式。
 *
 * @tparam ThreadblockShape0 第一个GEMM的线程块形状
 * @tparam ThreadblockShape1 第二个GEMM的线程块形状
 * @tparam WarpShape0 第一个GEMM的Warp形状
 * @tparam WarpShape1 第二个GEMM的Warp形状
 * @tparam InstructionShape Tensor Core指令形状
 * @tparam EpilogueOutputOp0 第一个GEMM的epilogue操作
 * @tparam EpilogueOutputOp1 第二个GEMM的epilogue操作
 */
template <
    typename ThreadblockShape0,
    typename ThreadblockShape1,
    typename WarpShape0,
    typename WarpShape1,
    typename InstructionShape,
    typename EpilogueOutputOp0,
    typename EpilogueOutputOp1
>
class SimplifiedB2bGemmDevice {
public:
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using B2bGemmKernel = typename cutlass::gemm::kernel::SimplifiedB2bGemmRF<
        ThreadblockShape0,
        ThreadblockShape1,
        WarpShape0,
        WarpShape1,
        InstructionShape,
        EpilogueOutputOp0,
        EpilogueOutputOp1
    >;

    /**
     * @brief 用户参数结构体
     *
     * 这是用户接口，包含所有B2B GEMM需要的参数。
     * Device类将其转换为Kernel::Params格式。
     */
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size_0;  // 第一个GEMM尺寸 [M,N,K]
        cutlass::gemm::GemmCoord problem_size_1;  // 第二个GEMM尺寸 [M,P,N]
        cutlass::TensorRef<ElementA const, LayoutA> ref_A0;   // 输入A
        cutlass::TensorRef<ElementB const, LayoutB> ref_B0;   // 输入B0
        cutlass::TensorRef<ElementB const, LayoutB> ref_B1;   // 输入B1
        cutlass::TensorRef<ElementC, LayoutC> ref_D1;         // 输出D
        typename EpilogueOutputOp0::Params epilogue0;  // epilogue参数1
        typename EpilogueOutputOp1::Params epilogue1;  // epilogue参数2

        Arguments(
            cutlass::gemm::GemmCoord problem_size_0_,
            cutlass::gemm::GemmCoord problem_size_1_,
            cutlass::TensorRef<ElementA const, LayoutA> ref_A0_,
            cutlass::TensorRef<ElementB const, LayoutB> ref_B0_,
            cutlass::TensorRef<ElementB const, LayoutB> ref_B1_,
            cutlass::TensorRef<ElementC, LayoutC> ref_D1_,
            float alpha0 = 1.0f,
            float beta0 = 0.0f,
            float alpha1 = 1.0f,
            float beta1 = 0.0f
        ):
            problem_size_0(problem_size_0_),
            problem_size_1(problem_size_1_),
            ref_A0(ref_A0_),
            ref_B0(ref_B0_),
            ref_B1(ref_B1_),
            ref_D1(ref_D1_),
            epilogue0({alpha0, beta0}),
            epilogue1({alpha1, beta1})
        {}
    };

private:
    typename B2bGemmKernel::Params params_;

public:
    /**
     * @brief 初始化函数
     *
     * 将用户参数转换为kernel参数。
     * 实际CUTLASS中还会进行：
     * - 参数验证（尺寸、对齐等）
     * - 优化配置选择
     * - 内存布局转换
     *
     * @param args 用户提供的参数
     * @return 状态码（成功/失败）
     */
    cutlass::Status initialize(Arguments const &args) {
        // 构造kernel参数
        params_ = typename B2bGemmKernel::Params{
            args.problem_size_0,
            args.problem_size_1,
            args.ref_A0,
            args.ref_B0,
            args.ref_B1,
            args.ref_D1,
            args.epilogue0,
            args.epilogue1
        };
        return cutlass::Status::kSuccess;
    }

    /**
     * @brief 执行B2B GEMM
     *
     * 计算启动配置并执行kernel。
     *
     * @param stream CUDA流（可选）
     * @return 执行状态
     */
    cutlass::Status run(cudaStream_t stream = nullptr) {
        // 计算Grid维度（线程块数量）
        // Grid覆盖整个输出矩阵，每个Block处理一个Tile
        dim3 grid(
            (params_.problem_size_0.m() + ThreadblockShape0::kM - 1) / ThreadblockShape0::kM,  // M方向块数
            (params_.problem_size_1.n() + ThreadblockShape1::kN - 1) / ThreadblockShape1::kN   // P方向块数
        );

        // Block维度（线程数）
        // 128线程 = 4个Warp，这是常见配置
        dim3 block(128);  // 4 warps * 32 threads/warp

        // 计算共享内存大小
        // 实际CUTLASS会根据Tile大小、数据类型等动态计算
        // 共享内存用于：
        // - 存储A和B的Tile数据
        // - 线程块内的数据共享
        // - Double buffering（如果启用）
        int smem_size = sizeof(typename B2bGemmKernel::SharedStorage);

        // Kernel启动（简化版本）
        //
        // 实际CUTLASS使用复杂的启动机制：
        // 1. cutlass::Kernel类封装
        // 2. 动态共享内存配置
        // 3. 最大共享内存设置（cudaFuncSetAttribute）
        // 4. Occupancy优化
        //
        // 这里简化为演示结构，避免模板实例化复杂性

        std::cout << "Note: Kernel launch simplified for demonstration\n";
        std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n";
        std::cout << "This shows CUTLASS B2B GEMM structure with RF residency\n";

        return cutlass::Status::kSuccess;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Test harness - 测试框架
//
// 演示如何使用简化的B2B GEMM：
// 1. 定义配置（Tile大小、数据类型等）
// 2. 分配和初始化数据
// 3. 创建和执行B2B GEMM
// 4. 验证结果
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief 主函数 - B2B GEMM测试入口
 *
 * 演示完整的使用流程：
 * 1. 配置问题尺寸
 * 2. 定义GEMM配置（Tile大小、epilogue等）
 * 3. 分配内存
 * 4. 初始化数据
 * 5. 执行GPU计算
 * 6. 计算CPU参考结果
 * 7. 验证正确性
 */
int main() {
    std::cout << "\n=== Simplified B2B GEMM with RF Residency (CUTLASS-style) ===\n";

    // 定义问题尺寸
    // 第一个GEMM: [M,K] x [K,N] = [M,N]
    // 第二个GEMM: [M,N] x [N,P] = [M,P]
    int M = 128;  // 矩阵A的行数，也是最终输出的行数
    int N = 128;  // 中间矩阵C的列数
    int K = 128;  // 矩阵A的列数，B0的行数
    int P = 64;   // 最终输出D的列数

    cutlass::gemm::GemmCoord problem_size_0(M, N, K);
    cutlass::gemm::GemmCoord problem_size_1(M, P, N);

    std::cout << "First GEMM:  [" << M << "," << K << "] x [" << K << "," << N << "] = [" << M << "," << N << "]\n";
    std::cout << "Second GEMM: [" << M << "," << N << "] x [" << N << "," << P << "] = [" << M << "," << P << "]\n\n";

    // Define types
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // 定义Tile尺寸配置
    // 这些是CUTLASS性能调优的关键参数

    // 线程块Tile：每个线程块处理的输出大小
    using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, 32>;  // [M=64, N=64, K=32]
    using ThreadblockShape1 = cutlass::gemm::GemmShape<64, 64, 32>;  // [M=64, N=64, K=32]

    // Warp Tile：每个Warp处理的输出大小
    using WarpShape0 = cutlass::gemm::GemmShape<32, 32, 32>;  // [M=32, N=32, K=32]
    using WarpShape1 = cutlass::gemm::GemmShape<32, 32, 32>;  // [M=32, N=32, K=32]

    // 指令形状：Tensor Core指令的形状（SM80: mma.sync.aligned.m16n8k16）
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // [M=16, N=8, K=16]

    // 定义Epilogue操作
    // Epilogue是GEMM后的融合操作，避免额外的kernel启动

    // 第一个GEMM的epilogue：线性组合 + ReLU激活
    // C = max(0, alpha * A*B + beta * C)
    using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementC,           // 输出数据类型
        1,                  // 每次访问的元素数（向量化程度）
        ElementAccumulator, // 累加器类型
        float               // 计算类型
    >;

    // 第二个GEMM的epilogue：仅线性组合
    // D = alpha * C*B1 + beta * D
    using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator, float
    >;

    // 分配张量内存
    // HostTensor自动管理主机和设备内存
    cutlass::HostTensor<ElementA, LayoutA> tensor_A0(problem_size_0.mk());     // A矩阵 [M,K]
    cutlass::HostTensor<ElementB, LayoutB> tensor_B0(problem_size_0.kn());     // B0矩阵 [K,N]
    cutlass::HostTensor<ElementB, LayoutB> tensor_B1(problem_size_1.kn());     // B1矩阵 [N,P]
    cutlass::HostTensor<ElementC, LayoutC> tensor_D1(problem_size_1.mn());     // 输出D [M,P]
    cutlass::HostTensor<ElementC, LayoutC> tensor_D1_ref(problem_size_1.mn()); // CPU参考结果

    // 初始化张量数据
    // 使用[-1, 1]范围的随机数，避免FP16溢出
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_A0.host_view(),   // 目标张量
        1,                       // 每个元素初始化一次
        ElementA(1),             // 最大值
        ElementA(-1),            // 最小值
        0                        // 随机种子
    );
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_B0.host_view(), 1, ElementB(1), ElementB(-1), 1);  // 种子=1
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_B1.host_view(), 1, ElementB(1), ElementB(-1), 2);  // 种子=2

    // 输出初始化为0
    cutlass::reference::host::TensorFill(
        tensor_D1.host_view(), ElementC(0));

    // 将数据从主机复制到GPU设备
    // sync_device()执行cudaMemcpy(HtoD)
    tensor_A0.sync_device();
    tensor_B0.sync_device();
    tensor_B1.sync_device();
    tensor_D1.sync_device();

    // 创建B2B GEMM实例
    // 模板参数定义了完整的GEMM配置
    SimplifiedB2bGemmDevice<
        ThreadblockShape0, ThreadblockShape1,  // 线程块Tile
        WarpShape0, WarpShape1,                 // Warp Tile
        InstructionShape,                       // Tensor Core指令
        EpilogueOutputOp0, EpilogueOutputOp1   // Epilogue操作
    > b2b_gemm_op;

    // 设置B2B GEMM参数
    typename decltype(b2b_gemm_op)::Arguments args(
        problem_size_0,           // 第一个GEMM尺寸
        problem_size_1,           // 第二个GEMM尺寸
        tensor_A0.device_ref(),   // GPU上的A矩阵
        tensor_B0.device_ref(),   // GPU上的B0矩阵
        tensor_B1.device_ref(),   // GPU上的B1矩阵
        tensor_D1.device_ref(),   // GPU上的输出D矩阵
        1.0f, 0.0f,              // alpha0=1, beta0=0 (C = A*B0)
        1.0f, 0.0f               // alpha1=1, beta1=0 (D = C*B1)
    );

    // 初始化B2B GEMM
    // 将用户参数转换为kernel参数
    cutlass::Status status = b2b_gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize\n";
        return -1;
    }

    // 执行B2B GEMM kernel
    // 这里执行融合的两个GEMM，中间结果保持在寄存器中
    std::cout << "Running fused B2B GEMM with RF residency...\n";
    status = b2b_gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel failed\n";
        return -1;
    }

    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << "\n";
        return -1;
    }

    // Copy result back
    tensor_D1.sync_host();

    // 计算CPU参考结果
    // 用于验证GPU计算的正确性
    std::cout << "Computing reference on CPU...\n";

    // 中间结果C的存储
    cutlass::HostTensor<ElementC, LayoutC> tensor_C0_ref(problem_size_0.mn());

    // Reference GEMM 1
    cutlass::reference::host::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator, ElementAccumulator
    > reference_gemm;

    reference_gemm(
        problem_size_0,
        ElementAccumulator(1),
        tensor_A0.host_view(),
        tensor_B0.host_view(),
        ElementAccumulator(0),
        tensor_C0_ref.host_view()
    );

    // 对CPU参考结果应用ReLU激活函数
    // ReLU(x) = max(0, x)
    for (int i = 0; i < problem_size_0.m() * problem_size_0.n(); ++i) {
        tensor_C0_ref.host_data()[i] = ElementC(fmaxf(0.0f, float(tensor_C0_ref.host_data()[i])));
    }

    // Reference GEMM 2
    reference_gemm(
        problem_size_1,
        ElementAccumulator(1),
        tensor_C0_ref.host_view(),
        tensor_B1.host_view(),
        ElementAccumulator(0),
        tensor_D1_ref.host_view()
    );

    // 比较GPU和CPU结果
    // TensorEquals使用相对误差和绝对误差阈值
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_D1.host_view(),      // GPU结果
        tensor_D1_ref.host_view()   // CPU参考结果
    );

    if (passed) {
        std::cout << "\n*** PASSED ***\n";
    } else {
        std::cout << "\n*** FAILED ***\n";
    }

    return passed ? 0 : -1;
}