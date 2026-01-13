// Test make_tiled_copy_C_atom to understand the intermediate layouts
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <iostream>
#include <iomanip>

using namespace cute;

template <class Layout>
void print_layout_info(const char* name, Layout const& layout) {
    std::cout << name << ": ";
    print(layout);
    std::cout << "\n";
}

template <int copy_V, class LayoutC_TV, class MmaTiler>
void test_copy_v(const char* title, LayoutC_TV const& layoutC_TV, MmaTiler const& mma_tiler) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Step 1: Truncate layoutC_TV
    std::cout << "【Step 1】截断 layoutC_TV 到 copy_V = " << copy_V << "\n";
    std::cout << std::string(50, '-') << "\n";

    auto truncate_shape = make_shape(size<0>(layoutC_TV), Int<copy_V>{});
    auto truncate_layout = make_layout(truncate_shape);

    auto layout_TV = composition(layoutC_TV, truncate_layout);
    std::cout << "layout_TV = composition(layoutC_TV, (" << size<0>(layoutC_TV) << ", " << copy_V << "))\n";
    std::cout << "         = "; print(layout_TV); std::cout << "\n";
    std::cout << "含义: (thr, val) -> mma_offset，共 " << size<0>(layout_TV) << " 个线程 × "
              << size<1>(layout_TV) << " 个值 = " << size(layout_TV) << " 元素\n\n";

    std::cout << "--- layout_TV 可视化 ---\n";
    print_layout(layout_TV);

    // Step 2: Compute tiler
    std::cout << "\n【Step 2】计算 tiler = (tiler_m, tiler_n)\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << "目的: 提取 layout_TV 在 M 和 N 方向的覆盖形状\n\n";

    auto mma_zeros = repeat_like(mma_tiler, Int<0>{});

    // For M dimension
    std::cout << "【2a】计算 tiler_m (M 方向覆盖)\n";
    auto stride_m = replace<0>(mma_zeros, Int<1>{});
    auto proj_m_layout = make_layout(mma_tiler, stride_m);
    std::cout << "  proj_m = make_layout(mma_tiler, (1,0)) = "; print(proj_m_layout); std::cout << "\n";
    std::cout << "  含义: (m,n) -> m，投影到 M 坐标\n\n";

    std::cout << "--- proj_m 可视化 (M,N) -> M坐标 ---\n";
    print_layout(proj_m_layout);

    auto tiler_m_raw = composition(proj_m_layout, layout_TV);
    std::cout << "\n  tiler_m_raw = composition(proj_m, layout_TV) = "; print(tiler_m_raw); std::cout << "\n";
    std::cout << "  含义: (thr, val) -> 该位置的 M 坐标\n\n";

    std::cout << "--- tiler_m_raw 可视化 (thr,val) -> M坐标 ---\n";
    print_layout(tiler_m_raw);

    auto tiler_m = filter(tiler_m_raw);
    std::cout << "\n  tiler_m = filter(tiler_m_raw) = "; print(tiler_m); std::cout << "\n";
    std::cout << "  filter 作用: 去除 stride=0 的模式（冗余维度），合并连续模式\n";
    std::cout << "  tiler_m_raw 有 " << size(tiler_m_raw) << " 个输入，但值域只有 " << size(tiler_m) << " 个不同输出\n";
    std::cout << "  结果: layout_TV 覆盖 M 坐标 [0, " << size(tiler_m)-1 << "]，stride=" << stride<0>(tiler_m) << "\n\n";

    // For N dimension
    std::cout << "【2b】计算 tiler_n (N 方向覆盖)\n";
    auto stride_n = replace<1>(mma_zeros, Int<1>{});
    auto proj_n_layout = make_layout(mma_tiler, stride_n);
    std::cout << "  proj_n = make_layout(mma_tiler, (0,1)) = "; print(proj_n_layout); std::cout << "\n";
    std::cout << "  含义: (m,n) -> n，投影到 N 坐标\n\n";

    std::cout << "--- proj_n 可视化 (M,N) -> N坐标 ---\n";
    print_layout(proj_n_layout);

    auto tiler_n_raw = composition(proj_n_layout, layout_TV);
    std::cout << "\n  tiler_n_raw = composition(proj_n, layout_TV) = "; print(tiler_n_raw); std::cout << "\n";
    std::cout << "  含义: (thr, val) -> 该位置的 N 坐标\n\n";

    std::cout << "--- tiler_n_raw 可视化 (thr,val) -> N坐标 ---\n";
    print_layout(tiler_n_raw);

    auto tiler_n = filter(tiler_n_raw);
    std::cout << "\n  tiler_n = filter(tiler_n_raw) = "; print(tiler_n); std::cout << "\n";
    std::cout << "  filter 作用: 去除 stride=0 的模式（冗余维度），合并连续模式\n";
    std::cout << "  tiler_n_raw 有 " << size(tiler_n_raw) << " 个输入，但值域只有 " << size(tiler_n) << " 个不同输出\n";
    std::cout << "  结果: layout_TV 覆盖 N 坐标 {";
    for (int i = 0; i < int(size(tiler_n)); ++i) {
        std::cout << tiler_n(i);
        if (i < int(size(tiler_n))-1) std::cout << ",";
    }
    std::cout << "}，stride=" << stride<0>(tiler_n) << "\n\n";

    // Combine as tiler tuple
    auto tiler = make_tuple(tiler_m, tiler_n);
    std::cout << "【2c】组合 tiler = (tiler_m, tiler_n)\n";
    std::cout << "  tiler = ("; print(tiler_m); std::cout << ", "; print(tiler_n); std::cout << ")\n";
    std::cout << "  tiler 是 tuple<Layout, Layout>，不是单个 2D Layout！\n\n";

    // 可视化 tiler_m
    std::cout << "--- tiler_m 可视化: m_idx -> M坐标 ---\n";
    std::cout << "tiler_m = "; print(tiler_m); std::cout << "\n";
    std::cout << "m_idx:  ";
    for (int m = 0; m < int(size(tiler_m)); ++m) {
        std::cout << std::setw(2) << m << " ";
    }
    std::cout << "\nM坐标:  ";
    for (int m = 0; m < int(size(tiler_m)); ++m) {
        std::cout << std::setw(2) << tiler_m(m) << " ";
    }
    std::cout << "\n\n";

    // 可视化 tiler_n
    std::cout << "--- tiler_n 可视化: n_idx -> N坐标 ---\n";
    std::cout << "tiler_n = "; print(tiler_n); std::cout << "\n";
    std::cout << "n_idx:  ";
    for (int n = 0; n < int(size(tiler_n)); ++n) {
        std::cout << std::setw(2) << n << " ";
    }
    std::cout << "\nN坐标:  ";
    for (int n = 0; n < int(size(tiler_n)); ++n) {
        std::cout << std::setw(2) << tiler_n(n) << " ";
    }
    std::cout << "\n\n";

    // tiler 在 composition 中的作用
    std::cout << "--- tiler 在 composition 中的作用 ---\n";
    std::cout << "composition(make_layout(mma_tiler), tiler) 时:\n";
    std::cout << "  输入 (m_idx, n_idx) → (tiler_m(m_idx), tiler_n(n_idx)) = (M, N)\n";
    std::cout << "  然后 (M, N) → M + N * " << size<0>(mma_tiler) << " = mma_offset\n";

    // Coverage on MMA tile
    std::cout << "\n--- tiler 在 MMA tile (" << size<0>(mma_tiler) << "x" << size<1>(mma_tiler) << ") 上的覆盖 ---\n";
    std::cout << "X = 被覆盖, . = 未覆盖\n";
    std::cout << "     N:  ";
    for (int n = 0; n < int(size<1>(mma_tiler)); ++n) {
        std::cout << n << " ";
    }
    std::cout << "\n  M     ";
    for (int n = 0; n < int(size<1>(mma_tiler)); ++n) {
        std::cout << "--";
    }
    std::cout << "\n";
    for (int m = 0; m < int(size<0>(mma_tiler)); ++m) {
        std::cout << "  " << std::setw(2) << m << " |  ";
        for (int n = 0; n < int(size<1>(mma_tiler)); ++n) {
            bool covered = false;
            for (int im = 0; im < int(size(tiler_m)); ++im) {
                for (int in = 0; in < int(size(tiler_n)); ++in) {
                    if (int(tiler_m(im)) == m && int(tiler_n(in)) == n) {
                        covered = true;
                        break;
                    }
                }
                if (covered) break;
            }
            std::cout << (covered ? "X " : ". ");
        }
        std::cout << "\n";
    }

    // Step 3: Compute tile2mma
    std::cout << "\n【Step 3】计算 tile2mma\n";
    std::cout << std::string(50, '-') << "\n";

    auto mma_layout = make_layout(mma_tiler);
    auto tile2mma = composition(mma_layout, tiler);

    std::cout << "mma_layout = make_layout(mma_tiler) = "; print(mma_layout); std::cout << "\n";
    std::cout << "  含义: (M, N) -> M + N * " << size<0>(mma_tiler) << " = mma_offset\n\n";

    std::cout << "tile2mma = composition(mma_layout, tiler)\n";
    std::cout << "  当 rhs 是 tuple 时，composition 对每个维度分别处理:\n";
    std::cout << "    composition(mma_layout[0], tiler_m) × composition(mma_layout[1], tiler_n)\n";
    std::cout << "    = composition("; print(get<0>(mma_layout)); std::cout << ", "; print(tiler_m); std::cout << ")\n";
    std::cout << "    × composition("; print(get<1>(mma_layout)); std::cout << ", "; print(tiler_n); std::cout << ")\n";
    std::cout << "  结果: "; print(tile2mma); std::cout << "\n\n";

    std::cout << "tile2mma 含义:\n";
    std::cout << "  (m_idx, n_idx) -> mma_layout(tiler_m(m_idx), tiler_n(n_idx))\n";
    std::cout << "                 -> tiler_m(m_idx) + tiler_n(n_idx) * " << size<0>(mma_tiler) << "\n";
    std::cout << "  shape: (" << size<0>(tile2mma) << ", " << size<1>(tile2mma) << ")\n\n";

    std::cout << "--- tile2mma 可视化 ---\n";
    print_layout(tile2mma);

    // Step 4: Compute left_inverse and layout_tv
    std::cout << "\n【Step 4】计算 left_inverse(tile2mma) 和 layout_tv\n";
    std::cout << std::string(50, '-') << "\n";

    auto inv_tile2mma = left_inverse(tile2mma);
    std::cout << "inv_tile2mma = left_inverse(tile2mma) = "; print(inv_tile2mma); std::cout << "\n";
    std::cout << "含义: mma_offset -> tile_coord (归一化坐标)\n\n";

    std::cout << "--- left_inverse(tile2mma) ---\n";
    std::cout << "注意: 只有前 " << size(tile2mma) << " 个 offset 有效\n";
    if constexpr (rank(inv_tile2mma) == 2) {
        print_layout(inv_tile2mma);
    } else {
        std::cout << "  (rank != 2, 无法用 print_layout 可视化)\n";
        std::cout << "  "; print(inv_tile2mma); std::cout << "\n";
    }

    auto layout_tv = composition(inv_tile2mma, layout_TV);
    std::cout << "\n【最终结果】layout_tv = composition(inv_tile2mma, layout_TV)\n";
    std::cout << "  = "; print(layout_tv); std::cout << "\n";
    std::cout << "含义: (thr, val) -> 归一化的 tile_coord\n";
    std::cout << "      输入: " << size<0>(layout_tv) << " 线程 × " << size<1>(layout_tv) << " 值\n";
    std::cout << "      输出: [0, " << size(layout_tv)-1 << "] 的归一化坐标\n\n";

    std::cout << "--- layout_tv (最终输出) 可视化 ---\n";
    print_layout(layout_tv);
}

int main() {
    std::cout << "=== Test make_tiled_copy_C_atom internals ===\n";
    std::cout << "理解 make_tiled_copy_C_atom 如何计算 layout_tv\n\n";

    // Use a simple SM80 MMA as example
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;

    auto tiled_mma = make_tiled_mma(MMA_Atom{},
                                     Layout<Shape<_1, _1, _1>>{},
                                     Tile<_16, _8, _16>{});

    std::cout << "=== 配置信息 ===\n";
    print(tiled_mma);
    std::cout << "\n";

    auto layoutC_TV = tiled_mma.get_layoutC_TV();
    std::cout << "\nlayoutC_TV: "; print(layoutC_TV); std::cout << "\n";
    std::cout << "  线程数: " << size<0>(layoutC_TV) << "\n";
    std::cout << "  每线程值数: " << size<1>(layoutC_TV) << "\n";
    std::cout << "  总元素: " << size(layoutC_TV) << "\n\n";

    auto mma_tiler = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma));
    std::cout << "mma_tiler (M x N): "; print(mma_tiler); std::cout << "\n";
    std::cout << "总元素: " << size(mma_tiler) << "\n";

    std::cout << "\n--- 原始 layoutC_TV 可视化 ---\n";
    print_layout(layoutC_TV);

    // Test different copy_V values
    test_copy_v<1>("copy_V = 1 (每线程1个值)", layoutC_TV, mma_tiler);
    test_copy_v<2>("copy_V = 2 (每线程2个值)", layoutC_TV, mma_tiler);
    test_copy_v<4>("copy_V = 4 (每线程4个值，完整)", layoutC_TV, mma_tiler);

    return 0;
}
