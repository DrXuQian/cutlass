#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== 静态类型 vs 动态类型初始化对比 ===\n\n";

    // ============================================================
    // 1. 静态类型：编译期完全确定，用 {} 初始化
    // ============================================================
    std::cout << "【静态类型 - 用 {} 初始化】\n";
    std::cout << "这些类型的所有信息在编译期就已确定\n\n";

    // 静态 Shape
    auto static_shape = Shape<_16, _8, _4>{};  // 编译期常量 (16, 8, 4)
    std::cout << "Shape<_16, _8, _4>{}  = " << static_shape << "\n";

    // 静态 Stride
    auto static_stride = Stride<_1, _16, _128>{};  // 编译期常量
    std::cout << "Stride<_1, _16, _128>{} = " << static_stride << "\n";

    // 静态 Layout（shape 和 stride 都是编译期常量）
    auto static_layout = Layout<Shape<_16, _8>, Stride<_1, _16>>{};
    std::cout << "Layout<Shape<_16,_8>, Stride<_1,_16>>{} = " << static_layout << "\n";

    // TiledMMA 的 LayoutC_TV - 完全静态
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_1, _1, _1>>>;

    // 这里用 {} 是因为 TiledMMA 的所有参数都是模板参数（编译期常量）
    TiledMMA tiled_mma{};  // 无需运行时参数

    auto layout_c_tv = tiled_mma.get_layoutC_TV();
    std::cout << "\nTiledMMA{}                            - 无需参数\n";
    std::cout << "tiled_mma.get_layoutC_TV() = " << layout_c_tv << "\n";

    // ============================================================
    // 2. 动态类型：需要运行时参数
    // ============================================================
    std::cout << "\n【动态类型 - 需要运行时参数】\n";
    std::cout << "这些类型需要在运行时提供具体数值\n\n";

    // 动态 Shape - 需要传入具体数值
    int m = 64, n = 32;
    auto dynamic_shape = make_shape(m, n);  // 运行时确定
    std::cout << "make_shape(64, 32)     = " << dynamic_shape << "\n";

    // 动态 Stride
    auto dynamic_stride = make_stride(1, m);  // stride 依赖于 m
    std::cout << "make_stride(1, 64)     = " << dynamic_stride << "\n";

    // 动态 Layout
    auto dynamic_layout = make_layout(dynamic_shape, dynamic_stride);
    std::cout << "make_layout(shape, stride) = " << dynamic_layout << "\n";

    // Tensor - 需要指针 + layout
    float data[64];
    auto tensor = make_tensor(&data[0], make_shape(8, 8), make_stride(1, 8));
    std::cout << "\nmake_tensor(ptr, shape, stride) - 需要 ptr + shape + stride\n";
    std::cout << "tensor.layout() = " << tensor.layout() << "\n";

    // ============================================================
    // 3. 混合类型：部分静态、部分动态
    // ============================================================
    std::cout << "\n【混合类型 - 部分静态部分动态】\n\n";

    // 静态 shape + 动态 stride
    auto mixed_layout = make_layout(Shape<_16, _8>{}, make_stride(1, m));
    std::cout << "Shape<_16,_8>{} + make_stride(1, m)\n";
    std::cout << "mixed_layout = " << mixed_layout << "\n";

    // ============================================================
    // 4. 为什么 LayoutC_TV 用 {} ？
    // ============================================================
    std::cout << "\n【为什么 TiledMMA::get_layoutC_TV() 返回的类型用 {} 初始化】\n\n";

    // 看看返回类型
    using LayoutC_TV_Type = decltype(tiled_mma.get_layoutC_TV());
    std::cout << "LayoutC_TV 类型: Layout<...静态Shape..., ...静态Stride...>\n";
    std::cout << "因为 MMA 的 tile 大小是编译期常量 (16x8x16)\n";
    std::cout << "所以 thread 到 value 的映射也是编译期完全确定的\n";
    std::cout << "无需任何运行时参数，用 {} 即可构造\n";

    // ============================================================
    // 5. 对比总结
    // ============================================================
    std::cout << "\n=== 总结 ===\n";
    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ 初始化方式          │ 使用场景                              │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Type{}              │ 所有模板参数都是编译期常量 (_N)       │\n";
    std::cout << "│ make_xxx(args...)   │ 需要运行时数值（变量、指针等）        │\n";
    std::cout << "│ 混合                │ 部分静态 + 部分动态                   │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";

    std::cout << "\n编译期常量例子：\n";
    std::cout << "  _1{}, _16{}, _128{}          - IntegralConstant\n";
    std::cout << "  Shape<_16, _8>{}             - 静态 Shape\n";
    std::cout << "  Stride<_1, _16>{}            - 静态 Stride\n";
    std::cout << "  Layout<...>{}                - 完全静态 Layout\n";
    std::cout << "  TiledMMA<...>{}              - MMA 配置\n";

    std::cout << "\n运行时参数例子：\n";
    std::cout << "  make_shape(m, n)             - m, n 是 int 变量\n";
    std::cout << "  make_stride(1, lda)          - lda 是 leading dimension\n";
    std::cout << "  make_tensor(ptr, layout)     - ptr 是运行时指针\n";

    return 0;
}
