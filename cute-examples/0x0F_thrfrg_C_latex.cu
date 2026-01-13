#include <cstdio>
#include <fstream>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/util/print_latex.hpp>

using namespace cute;

// CLayout_64xN definition (from CUTLASS)
template <int N>
using CLayout_64xN = Layout<Shape <Shape <  _4, _8, _4>, Shape < _2, _2, Int<N/8>>>,
                            Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;

using CLayout_64x16 = CLayout_64xN<16>;

// Helper to redirect stdout to file
class StdoutRedirect {
    FILE* old_stdout;
    FILE* file;
public:
    StdoutRedirect(const char* filename) {
        file = fopen(filename, "w");
        old_stdout = stdout;
        stdout = file;
    }
    ~StdoutRedirect() {
        stdout = old_stdout;
        fclose(file);
    }
};

int main(int argc, char** argv) {

    printf("============================================================\n");
    printf("    Generating LaTeX files for thrfrg_C step-by-step\n");
    printf("============================================================\n\n");

    //==========================================================================
    // Parameters
    //==========================================================================

    using AtomShapeM = Int<64>;
    using AtomShapeN = Int<16>;
    using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;
    using AtomLayoutC_TV = CLayout_64x16;

    //==========================================================================
    // Step 0: Original layout (8x8 for visualization)
    //==========================================================================

    printf("=== Step 0: Generating original 8x8 layout ===\n");
    {
        StdoutRedirect redirect("step0_original_8x8.tex");
        auto layout = make_layout(make_shape(Int<8>{}, Int<8>{}));
        print_latex(layout);
    }
    printf("  -> step0_original_8x8.tex\n");

    //==========================================================================
    // Step 1: logical_divide result
    //==========================================================================

    printf("=== Step 1: Generating logical_divide result ===\n");
    {
        StdoutRedirect redirect("step1_logical_divide.tex");
        auto layout_2d = make_layout(make_shape(Int<8>{}, Int<8>{}));
        auto tile_2d = make_tile(make_layout(Int<4>{}), make_layout(Int<2>{}));
        auto divided = logical_divide(layout_2d, tile_2d);
        print_latex(divided);
    }
    printf("  -> step1_logical_divide.tex\n");

    //==========================================================================
    // Step 2: zipped_divide result
    //==========================================================================

    printf("=== Step 2: Generating zipped_divide result ===\n");
    {
        StdoutRedirect redirect("step2_zipped_divide.tex");
        auto layout_2d = make_layout(make_shape(Int<8>{}, Int<8>{}));
        auto tile_2d = make_tile(make_layout(Int<4>{}), make_layout(Int<2>{}));
        auto zipped = zipped_divide(layout_2d, tile_2d);
        print_latex(zipped);
    }
    printf("  -> step2_zipped_divide.tex\n");

    //==========================================================================
    // Step 3: AtomLayoutC_TV (CLayout_64x16) - 16x16 subset for visibility
    //==========================================================================

    printf("=== Step 3: Generating AtomLayoutC_TV (16x8 subset) ===\n");
    {
        StdoutRedirect redirect("step3_atom_layout_tv.tex");
        // Use smaller layout for visualization: 16 threads x 8 values
        using SmallLayout = Layout<Shape<_16, _8>, Stride<_1, _16>>;
        print_latex(SmallLayout{});
    }
    printf("  -> step3_atom_layout_tv.tex\n");

    //==========================================================================
    // Step 4: Full CLayout_64x16 visualization (first 32 threads)
    //==========================================================================

    printf("=== Step 4: Generating CLayout_64x16 (32 threads x 8 values) ===\n");
    {
        StdoutRedirect redirect("step4_clayout_64x16.tex");
        // CLayout_64x16 is 128 threads x 8 values
        // For visualization, show structure
        auto atom_tv = AtomLayoutC_TV{};

        // Print as 2D layout showing thread x value -> offset
        printf("%% CLayout_64x16: "); print(atom_tv); printf("\n");
        printf("\\documentclass[convert]{standalone}\n"
               "\\usepackage{tikz}\n\n"
               "\\begin{document}\n"
               "\\begin{tikzpicture}[x={(0cm,-0.6cm)},y={(0.8cm,0cm)},every node/.style={minimum size=0.6cm, outer sep=0pt, font=\\tiny}]\n\n");

        // Show first 32 threads x 8 values
        for (int t = 0; t < 32; ++t) {
            for (int v = 0; v < 8; ++v) {
                int offset = atom_tv(t, v);
                int m = offset % 64;
                int n = offset / 64;
                printf("\\node[fill=black!%d] at (%d,%d) {%d};\n",
                       (t % 8) * 10 + 10, t, v, offset);
            }
        }

        printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (32,8);\n");
        printf("\\node at (-1, 3.5) {Thread};\n");
        printf("\\node at (16, -1.5) {Value};\n");

        printf("\\end{tikzpicture}\n"
               "\\end{document}\n");
    }
    printf("  -> step4_clayout_64x16.tex\n");

    //==========================================================================
    // Step 5: Thread 0's values in 64x16 matrix
    //==========================================================================

    printf("=== Step 5: Generating Thread 0's values in 64x16 matrix ===\n");
    {
        StdoutRedirect redirect("step5_thread0_positions.tex");
        auto atom_tv = AtomLayoutC_TV{};

        printf("%% Thread 0's 8 values in 64x16 matrix\n");
        printf("\\documentclass[convert]{standalone}\n"
               "\\usepackage{tikz}\n\n"
               "\\begin{document}\n"
               "\\begin{tikzpicture}[x={(0cm,-0.3cm)},y={(0.5cm,0cm)},every node/.style={minimum size=0.3cm, outer sep=0pt, font=\\tiny}]\n\n");

        // Draw 16x16 grid (showing k=0..15, m=0..15 subset)
        for (int m = 0; m < 16; ++m) {
            for (int n = 0; n < 16; ++n) {
                printf("\\node[fill=white] at (%d,%d) {};\n", m, n);
            }
        }

        // Highlight Thread 0's values
        for (int v = 0; v < 8; ++v) {
            int offset = atom_tv(0, v);
            int m = offset % 64;
            int n = offset / 64;
            if (m < 16 && n < 16) {
                printf("\\node[fill=red!50] at (%d,%d) {V%d};\n", m, n, v);
            }
        }

        printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (16,16);\n");
        printf("\\node at (-2, 8) {m};\n");
        printf("\\node at (8, -2) {k};\n");

        printf("\\end{tikzpicture}\n"
               "\\end{document}\n");
    }
    printf("  -> step5_thread0_positions.tex\n");

    //==========================================================================
    // Step 6: zipped_divide for thrfrg_C (smaller example)
    //==========================================================================

    printf("=== Step 6: Generating zipped_divide for 16x16 matrix ===\n");
    {
        StdoutRedirect redirect("step6_zipped_divide_16x16.tex");
        auto ctensor = make_layout(make_shape(Int<16>{}, Int<16>{}));
        auto c_tile = make_tile(make_layout(Int<8>{}), make_layout(Int<4>{}));
        auto c_tensor = zipped_divide(ctensor, c_tile);
        print_latex(c_tensor);
    }
    printf("  -> step6_zipped_divide_16x16.tex\n");

    //==========================================================================
    // Print summary of layouts
    //==========================================================================

    printf("\n============================================================\n");
    printf("                    Layout Summary\n");
    printf("============================================================\n\n");

    auto ctensor = make_layout(make_shape(Int<128>{}, Int<128>{}));
    printf("Step 0: ctensor = "); print(ctensor); printf("\n");

    auto c_tile = make_tile(make_layout(AtomShapeM{}), make_layout(AtomShapeN{}));
    printf("Step 2: c_tile = "); print(c_tile); printf("\n");

    auto c_tensor = zipped_divide(ctensor, c_tile);
    printf("Step 2: zipped_divide result = "); print(c_tensor); printf("\n");
    printf("  Shape: "); print(shape(c_tensor)); printf("\n");

    auto tv_tensor = c_tensor.compose(AtomLayoutC_TV{}, _);
    printf("Step 3: compose with AtomLayoutC_TV = "); print(tv_tensor); printf("\n");
    printf("  Shape: "); print(shape(tv_tensor)); printf("\n");

    auto AtomThrID = Layout<Int<128>>{};
    auto thr_layout_vmnk = tiled_product(AtomThrID, AtomLayoutMNK{});
    printf("Step 4: thr_layout_vmnk = "); print(thr_layout_vmnk); printf("\n");

    auto thr_tile = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                           make_layout(size<2>(thr_layout_vmnk))));
    printf("Step 4: thr_tile = "); print(thr_tile); printf("\n");

    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);
    printf("Step 4: final thr_tensor = "); print(thr_tensor); printf("\n");
    printf("  Shape: "); print(shape(thr_tensor)); printf("\n");

    printf("\n============================================================\n");
    printf("    LaTeX files generated. Compile with:\n");
    printf("    pdflatex -shell-escape stepX_*.tex\n");
    printf("============================================================\n");

    return 0;
}
