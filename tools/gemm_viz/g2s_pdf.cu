// G2S (gmem -> smem) visualization to PDF, derived directly from CUTE layouts.
//
// This tool targets the HGEMM TN tutorial in:
//   examples/cute/tutorial/sgemm_sm80.cu
//
// Output:
//  - per-stage multi-page PDFs (easy to scroll)
//  - per-step single-page PDFs (easy to convert into GIF)
//
// Aesthetics: generate PDFs via LaTeX/TikZ (print_latex-style), in landscape layout.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/util/print_latex.hpp>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

struct Options {
  // If not set explicitly, we default to "<exe_dir>/out_g2s" (see parse_opts).
  std::string outdir{};
  double cell_mm{1.8};   // cell size in mm (TikZ units)
  double gap_mm{10.0};
  double margin_x_mm{10.0};
  bool emit_stage_pdf{true};
  bool emit_step_pdfs{true};
  bool emit_layout_pdfs{true};
  bool emit_combined_pdf{true};
  int pipe{-1};          // -1 = all pipes, else [0..kPipe-1]
  int k_tile{0};         // preset only supports 0 for now
};

static Options parse_opts(int argc, char** argv) {
  Options o;
  {
    fs::path exe = (argc > 0 && argv && argv[0]) ? fs::path(argv[0]) : fs::path("g2s_pdf");
    fs::path exe_dir = exe.has_parent_path() ? exe.parent_path() : fs::path(".");
    o.outdir = (exe_dir / "out_g2s").string();
  }
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--outdir" && i + 1 < argc) {
      o.outdir = argv[++i];
      continue;
    }
    if (a == "--cell-mm" && i + 1 < argc) {
      o.cell_mm = std::max(0.2, std::atof(argv[++i]));
      continue;
    }
    if (a == "--gap-mm" && i + 1 < argc) {
      o.gap_mm = std::max(0.0, std::atof(argv[++i]));
      continue;
    }
    if (a == "--no-stage") {
      o.emit_stage_pdf = false;
      continue;
    }
    if (a == "--no-steps") {
      o.emit_step_pdfs = false;
      continue;
    }
    if (a == "--no-layouts") {
      o.emit_layout_pdfs = false;
      continue;
    }
    if (a == "--no-combined") {
      o.emit_combined_pdf = false;
      continue;
    }
    if (a == "--k-tile" && i + 1 < argc) {
      o.k_tile = std::max(0, std::atoi(argv[++i]));
      continue;
    }
    if (a == "--pipe" && i + 1 < argc) {
      o.pipe = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--help" || a == "-h") {
      std::printf(
          "g2s_pdf (preset=sgemm_sm80)\n"
          "  --outdir <dir>     Output directory (default <exe_dir>/out_g2s)\n"
          "  --cell-mm <mm>     Cell size in mm (default 1.8)\n"
          "  --gap-mm <mm>      Gap between panels (default 10)\n"
          "  --k-tile <i>       Which k_tile to visualize (default 0, preset supports only 0)\n"
          "  --pipe <i>         Which pipe stage to visualize (default -1 = all)\n"
          "  --no-layouts       Skip generating copyA/copyB layout PDFs\n"
          "  --no-combined      Skip generating interleaved A/B stage PDFs\n"
          "  --no-stage         Skip per-stage multi-page PDFs\n"
          "  --no-steps         Skip per-step PDFs\n");
      std::exit(0);
    }
  }
  return o;
}

static std::string shell_quote(std::string const& s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

static void run_cmd(std::string const& cmd) {
  int rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("Command failed: " + cmd);
  }
}

static void run_pdflatex(fs::path const& workdir, std::string const& tex, std::string const& jobname = {}) {
  std::ostringstream cmd;
  cmd << "cd " << shell_quote(workdir.string()) << " && "
      << "pdflatex -interaction=batchmode -halt-on-error -file-line-error ";
  if (!jobname.empty()) {
    cmd << "-jobname " << shell_quote(jobname) << " ";
  }
  cmd << shell_quote(tex) << " >/dev/null";
  run_cmd(cmd.str());
}

#if !defined(_WIN32)
class ScopedStdoutRedirect {
 public:
  explicit ScopedStdoutRedirect(std::string const& path) {
    std::fflush(stdout);
    saved_fd_ = ::dup(fileno(stdout));
    if (saved_fd_ < 0) {
      ok_ = false;
      return;
    }
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      ok_ = false;
      return;
    }
    if (::dup2(fd, fileno(stdout)) < 0) {
      ok_ = false;
    }
    ::close(fd);
  }

  ScopedStdoutRedirect(ScopedStdoutRedirect const&) = delete;
  ScopedStdoutRedirect& operator=(ScopedStdoutRedirect const&) = delete;

  ~ScopedStdoutRedirect() {
    if (saved_fd_ < 0) return;
    std::fflush(stdout);
    (void)::dup2(saved_fd_, fileno(stdout));
    ::close(saved_fd_);
  }

  bool ok() const { return ok_; }

 private:
  int saved_fd_{-1};
  bool ok_{true};
};

template <class Fn>
static void dump_stdout_to_file(std::string const& path, Fn&& fn) {
  ScopedStdoutRedirect redirect(path);
  if (!redirect.ok()) {
    throw std::runtime_error("Failed to redirect stdout to " + path);
  }
  fn();
  std::fflush(stdout);
}
#endif

struct RGB {
  int r{0};
  int g{0};
  int b{0};
};

static RGB hsl_to_rgb_u8(double h_deg, double s, double l) {
  double h = h_deg / 360.0;
  while (h < 0) h += 1.0;
  while (h >= 1) h -= 1.0;

  auto hue2rgb = [](double p, double q, double t) {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
  };

  double r = l, g = l, b = l;
  if (s != 0.0) {
    double q = (l < 0.5) ? (l * (1 + s)) : (l + s - l * s);
    double p = 2 * l - q;
    r = hue2rgb(p, q, h + 1.0 / 3.0);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1.0 / 3.0);
  }
  auto to_u8 = [](double x) {
    int v = int(x * 255.0 + 0.5);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return v;
  };
  return RGB{to_u8(r), to_u8(g), to_u8(b)};
}

static int smem_bank_from_elem_offset(int elem_offset, int elem_bytes) {
  int byte_off = elem_offset * elem_bytes;
  return ((byte_off / 4) & 31); // 32 banks, 4B each
}

struct ThreadBox {
  int tid{0};
  int warp{0};
  int lane{0};
  int row_min{1 << 30};  // inclusive
  int row_max{-1};       // inclusive
  int col_min{1 << 30};  // inclusive
  int col_max{-1};       // inclusive

  bool valid() const { return row_max >= row_min && col_max >= col_min; }
};

struct TwoPanelLayout {
  double paper_w_mm{420.0}; // A3 landscape
  double paper_h_mm{297.0};
  double cell_mm{1.8};
  double gap_mm{10.0};
  double margin_x_mm{10.0};
  double margin_y_mm{50.0}; // top margin (panels start at y0)

  int rows{128};
  int cols{64};

  double panel_w() const { return cols * cell_mm; }
  double panel_h() const { return rows * cell_mm; }

  double y0() const { return margin_y_mm; }

  double g_x() const { return margin_x_mm; }
  double s_x() const { return g_x() + panel_w() + gap_mm; }
  double inset_x() const { return s_x() + panel_w() + gap_mm; }
  double inset_w() const { return std::max(0.0, paper_w_mm - inset_x() - margin_x_mm); }
};

static void write_text_file(fs::path const& path, std::string const& text) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Failed to write " + path.string());
  out << text;
}

static void force_pdf14_in_tex(fs::path const& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to read " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string text = ss.str();
  if (text.find("\\pdfminorversion") != std::string::npos) return;
  std::string insert = "\\pdfminorversion=4\n\\pdfobjcompresslevel=0\n\\pdfcompresslevel=9\n";
  std::string marker = "\\begin{document}";
  std::size_t pos = text.find(marker);
  if (pos == std::string::npos) return;
  text.insert(pos, insert);
  write_text_file(path, text);
}

static std::string latex_preamble_landscape_a3() {
  return R"(\documentclass[a3paper,landscape]{article}
\usepackage[margin=0mm]{geometry}
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{pdfpages}
\pdfminorversion=4
\pdfobjcompresslevel=0
\pdfcompresslevel=9
\pagestyle{empty}
\begin{document}
)";
}

static std::string latex_footer() { return "\\end{document}\n"; }

static std::string define_colors_block() {
  // Warp colors matching tools/gemm_viz/app.js.
  std::ostringstream oss;
  oss << "\\definecolor{warp0}{RGB}{59,130,246}\n";
  oss << "\\definecolor{warp1}{RGB}{139,92,246}\n";
  oss << "\\definecolor{warp2}{RGB}{236,72,153}\n";
  oss << "\\definecolor{warp3}{RGB}{249,115,22}\n";
  // Bank colors: 32 distinct hues.
  for (int b = 0; b < 32; ++b) {
    double h = (double(b) / 32.0) * 360.0;
    // Soft palette: SMEM bank coloring should aid pattern recognition, not overpower highlights.
    RGB c = hsl_to_rgb_u8(h, 0.45, 0.87);
    oss << "\\definecolor{bank" << b << "}{RGB}{" << c.r << "," << c.g << "," << c.b << "}\n";
  }
  return oss.str();
}

template <class SmemLayout>
static std::string make_background_tex(TwoPanelLayout const& pl,
                                       SmemLayout const& s_layout,
                                       int pipe,
                                       int elem_bytes,
                                       char const* operand_name,
                                       char const* row_name,
                                       std::string const& copy_pdf_name) {
  std::ostringstream tex;
  tex << latex_preamble_landscape_a3();
  tex << define_colors_block();
  tex << "\\def\\Cell{" << pl.cell_mm << "}\n";

  // In TikZ overlay mode with margin=0, the origin behaves like the page's top-left.
  // We flip the y-axis so y increases downward (more intuitive for matrix row indices).
  tex << "\\begin{tikzpicture}[remember picture,overlay,x=1mm,y=-1mm]\n";

  const double y0 = pl.y0();
  const double pw = pl.panel_w();
  const double ph = pl.panel_h();

  // Panel labels
  tex << "  \\node[anchor=south west] at (" << pl.g_x() << "," << (y0 - 6.0)
      << ") {\\Large\\texttt{g" << operand_name << " (" << row_name << ",k)  CTA tile}};\n";
  tex << "  \\node[anchor=south west] at (" << pl.s_x() << "," << (y0 - 6.0)
      << ") {\\Large\\texttt{s" << operand_name << " (" << row_name << ",k)  bank-colored}};\n";

  // Smem bank background fills (no per-cell text).
  for (int r = 0; r < pl.rows; ++r) {
    for (int c = 0; c < pl.cols; ++c) {
      int off = int(s_layout(r, c, pipe));
      int bank = smem_bank_from_elem_offset(off, elem_bytes);
      double x = pl.s_x() + c * pl.cell_mm;
      double y = y0 + r * pl.cell_mm;
      tex << "  \\fill[fill=bank" << bank << "] (" << x << "," << y << ") rectangle ++(\\Cell,\\Cell);\n";
    }
  }

  // Grid lines + borders
  auto emit_grid = [&](double x0) {
    const double minor = pl.cell_mm;
    const double major = pl.cell_mm * 8.0;
    tex << "  \\draw[black!70,line width=0.20mm] (" << x0 << "," << y0 << ") rectangle ++(" << pw << "," << ph << ");\n";
    tex << "  \\draw[black!10,line width=0.02mm] (" << x0 << "," << y0 << ") grid[step=" << minor << "] ("
        << (x0 + pw) << "," << (y0 + ph) << ");\n";
    tex << "  \\draw[black!30,line width=0.08mm] (" << x0 << "," << y0 << ") grid[step=" << major << "] ("
        << (x0 + pw) << "," << (y0 + ph) << ");\n";
  };
  emit_grid(pl.g_x());
  emit_grid(pl.s_x());

  // Warp legend (top-right)
  {
    const double box_w = 14.0;
    const double box_h = 6.0;
    const double gap = 2.0;
    const double total_w = 4 * box_w + 3 * gap;
    const double x0 = pl.paper_w_mm - pl.margin_x_mm - total_w;
    const double y = 8.0;
    for (int w = 0; w < 4; ++w) {
      double x = x0 + w * (box_w + gap);
      tex << "  \\fill[fill=warp" << w << "] (" << x << "," << y << ") rectangle ++(" << box_w << "," << box_h << ");\n";
      tex << "  \\node[anchor=center,text=white] at (" << (x + box_w * 0.5) << "," << (y + box_h * 0.5)
          << ") {\\scriptsize\\texttt{W" << w << "}};\n";
    }
  }

  // Bank legend (top, left)
  {
    const double box_w = 6.0;
    const double box_h = 4.5;
    const double gap = 0.6;
    const double x0 = pl.margin_x_mm;
    const double y = 18.0;
    tex << "  \\node[anchor=south west] at (" << x0 << "," << y << ") {\\small\\texttt{bank:}};\n";
    double bx = x0 + 16.0;
    for (int b = 0; b < 32; ++b) {
      double x = bx + b * (box_w + gap);
      tex << "  \\fill[fill=bank" << b << "] (" << x << "," << y << ") rectangle ++(" << box_w << "," << box_h << ");\n";
      tex << "  \\draw[black!40,line width=0.05mm] (" << x << "," << y << ") rectangle ++(" << box_w << "," << box_h << ");\n";
      tex << "  \\node[anchor=center,black] at (" << (x + box_w * 0.5) << "," << (y + box_h * 0.5)
          << ") {\\tiny\\texttt{" << b << "}};\n";
    }
  }

  // Axis ticks
  auto emit_ticks = [&](double x0) {
    const double tick_x = x0 - 2.0;
    for (int r = 0; r < pl.rows; r += 16) {
      double y = y0 + r * pl.cell_mm + pl.cell_mm * 0.5;
      tex << "  \\node[anchor=east] at (" << tick_x << "," << y << ") {\\scriptsize\\texttt{" << row_name << "=" << r << "}};\n";
    }
    const double tick_y = y0 + ph + 4.5;
    for (int c = 0; c < pl.cols; c += 16) {
      double x = x0 + c * pl.cell_mm + pl.cell_mm * 0.5;
      tex << "  \\node[anchor=north] at (" << x << "," << tick_y << ") {\\scriptsize\\texttt{" << c << "}};\n";
    }
    tex << "  \\node[anchor=north west] at (" << x0 << "," << (y0 + ph + 14.0) << ") {\\scriptsize\\texttt{k}};\n";
  };
  emit_ticks(pl.g_x());
  emit_ticks(pl.s_x());

  // Optional CUTE layout inset (e.g., copyA.pdf / copyB.pdf) in the right margin.
  if (!copy_pdf_name.empty() && pl.inset_w() > 20.0) {
    const double x = pl.inset_x();
    const double y = y0; // align with top of panels
    const double w = std::min(pl.inset_w(), 150.0);
    tex << "  \\node[anchor=north west] at (" << x << "," << (y - 12.0) << ") {\\normalsize\\texttt{CUTE:}};\n";
    tex << "  \\node[anchor=north west,inner sep=0] at (" << x << "," << y << ") {\\includegraphics[width="
        << w << "mm]{" << copy_pdf_name << "}};\n";
  }

  tex << "\\end{tikzpicture}\n";
  tex << latex_footer();
  return tex.str();
}

static std::string make_stage_tex(TwoPanelLayout const& pl,
                                  char const* operand_name,
                                  char const* row_name,
                                  int pipe,
                                  int k_tile,
                                  int V,
                                  int cpy_m,
                                  int cpy_k,
                                  int steps_per_stage,
                                  std::vector<std::vector<ThreadBox>> const& boxes,
                                  std::string const& bg_pdf_name) {
  std::ostringstream tex;
  tex << latex_preamble_landscape_a3();
  tex << define_colors_block();
  tex << "\\def\\Cell{" << pl.cell_mm << "}\n";
  tex << "\\def\\TidFont{\\fontsize{3.8}{3.8}\\selectfont}\n";

  const double y0 = pl.y0();

  auto rect_xywh = [&](double x0, ThreadBox const& b) {
    double x = x0 + b.col_min * pl.cell_mm;
    double y = y0 + b.row_min * pl.cell_mm;
    double w = (b.col_max - b.col_min + 1) * pl.cell_mm;
    double h = (b.row_max - b.row_min + 1) * pl.cell_mm;
    return std::tuple<double,double,double,double>(x, y, w, h);
  };

  for (int step = 0; step < steps_per_stage; ++step) {
    int cm = (cpy_m > 0) ? (step % cpy_m) : 0;
    int ck = (cpy_m > 0) ? (step / cpy_m) : 0;

    tex << "\\begin{tikzpicture}[remember picture,overlay,x=1mm,y=-1mm]\n";
    tex << "  \\node[anchor=north west,inner sep=0] at (0,0) {\\includegraphics[width=\\paperwidth,height=\\paperheight]{"
        << bg_pdf_name << "}};\n";

    const double header_y = 8.0;
    tex << "  \\node[anchor=north west] at (" << pl.margin_x_mm << "," << header_y << ") {\\Large\\texttt{G2S "
        << operand_name << " (" << row_name << ",k)  pipe=" << pipe << "  k\\_tile=" << k_tile
        << "  step " << step << "/" << (steps_per_stage - 1)
        << "  (cm=" << cm << ", ck=" << ck << ")"
        << "  (cpy\\_m=" << cpy_m << ", cpy\\_k=" << cpy_k << ")"
        << "  (V=" << V << " half = " << (V * 2) << "B)}};\n";
    tex << "  \\node[anchor=north west] at (" << pl.margin_x_mm << "," << (header_y + 10.0)
        << ") {\\normalsize\\texttt{Overlay box = one thread cp.async. Text = tid.}};\n";

    for (auto const& b : boxes[step]) {
      if (!b.valid()) continue;
      std::string color = "warp" + std::to_string(b.warp & 3);
      auto [xg, yg, wg, hg] = rect_xywh(pl.g_x(), b);
      auto [xs, ys, ws, hs] = rect_xywh(pl.s_x(), b);

      // gmem: filled box
      tex << "  \\fill[fill=" << color << ",fill opacity=0.65] (" << xg << "," << yg << ") rectangle ++(" << wg << "," << hg << ");\n";
      tex << "  \\draw[draw=" << color << ",line width=0.25mm] (" << xg << "," << yg << ") rectangle ++(" << wg << "," << hg << ");\n";

      // smem: outline + light tint (bank background remains visible)
      tex << "  \\fill[fill=" << color << ",fill opacity=0.18] (" << xs << "," << ys << ") rectangle ++(" << ws << "," << hs << ");\n";
      tex << "  \\draw[draw=" << color << ",line width=0.30mm] (" << xs << "," << ys << ") rectangle ++(" << ws << "," << hs << ");\n";

      // tid labels at center of each box
      double cx_g = xg + wg * 0.5;
      double cy_g = yg + hg * 0.5;
      double cx_s = xs + ws * 0.5;
      double cy_s = ys + hs * 0.5;
      tex << "  \\node[anchor=center,font=\\TidFont,text=white] at (" << cx_g << "," << cy_g << ") {\\texttt{t" << b.tid << "}};\n";
      tex << "  \\node[anchor=center,font=\\TidFont,text=black] at (" << cx_s << "," << cy_s << ") {\\texttt{t" << b.tid << "}};\n";
    }

    tex << "\\end{tikzpicture}\n";
    if (step + 1 != steps_per_stage) tex << "\\newpage\n";
  }

  tex << latex_footer();
  return tex.str();
}

static std::string make_interleaved_stage_tex(std::string const& stageA_pdf,
                                              std::string const& stageB_pdf,
                                              int steps_per_stage) {
  std::ostringstream tex;
  tex << latex_preamble_landscape_a3();
  for (int step = 0; step < steps_per_stage; ++step) {
    tex << "\\includepdf[pages=" << (step + 1) << "]{" << stageA_pdf << "}\n";
    tex << "\\includepdf[pages=" << (step + 1) << "]{" << stageB_pdf << "}\n";
  }
  tex << latex_footer();
  return tex.str();
}

static std::string make_extract_page_tex(std::string const& stage_pdf, int page_1based) {
  std::ostringstream tex;
  tex << latex_preamble_landscape_a3();
  tex << "\\includepdf[pages=" << page_1based << "]{" << stage_pdf << "}\n";
  tex << latex_footer();
  return tex.str();
}

static void safe_rename(fs::path const& src, fs::path const& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  fs::remove(dst, ec);
  fs::rename(src, dst, ec);
  if (!ec) return;
  // Fallback: copy+remove
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) throw std::runtime_error("Failed to move " + src.string() + " -> " + dst.string() + ": " + ec.message());
  fs::remove(src, ec);
}

static void safe_copy(fs::path const& src, fs::path const& dst) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) throw std::runtime_error("Failed to copy " + src.string() + " -> " + dst.string() + ": " + ec.message());
}

static void generate(Options const& opt) {
  using namespace cute;
  using T = half_t;

  // Preset constants: match sgemm_sm80.cu gemm_tn() tutorial kernel.
  constexpr int kB_M = 128;
  constexpr int kB_N = 128;
  constexpr int kB_K = 64;
  constexpr int kPipe = 3;
  constexpr int kNumThreads = 128;
  constexpr int kElemBytes = 2; // half
  constexpr int kKTileCount = 1;

  if (opt.k_tile != 0 || opt.k_tile >= kKTileCount) {
    throw std::runtime_error("Preset currently supports only --k-tile 0");
  }

  TwoPanelLayout plA;
  plA.cell_mm = opt.cell_mm;
  plA.gap_mm = opt.gap_mm;
  plA.margin_x_mm = opt.margin_x_mm;
  plA.rows = kB_M;
  plA.cols = kB_K;

  TwoPanelLayout plB = plA;
  plB.rows = kB_N;

  auto check_fit = [&](TwoPanelLayout const& pl) {
    const double total_w = pl.s_x() + pl.panel_w() + pl.margin_x_mm;
    if (total_w > pl.paper_w_mm - 1e-6) {
      std::ostringstream msg;
      msg << "Layout does not fit on A3 landscape. Try smaller --cell-mm. "
          << "Needed width=" << total_w << "mm, paper width=" << pl.paper_w_mm << "mm.";
      throw std::runtime_error(msg.str());
    }
    if (pl.y0() < 0.0 || (pl.y0() + pl.panel_h() > pl.paper_h_mm - 1e-6)) {
      throw std::runtime_error("Layout too tall for A3 landscape. Try smaller --cell-mm.");
    }
  };
  check_fit(plA);
  check_fit(plB);

  fs::path outdir = opt.outdir;
  fs::path build = outdir / "_build";
  fs::create_directories(build);

  // === Layouts (smem swizzle): match sgemm_sm80.cu ===
  auto bM = Int<kB_M>{};
  auto bN = Int<kB_N>{};
  auto bK = Int<kB_K>{};
  auto bP = Int<kPipe>{};

  using SwizzleBase =
      Layout<Shape<_8, Shape<_8, _8>>,
             Stride<_8, Stride<_1, _64>>>;
  SwizzleBase swizzle_base{};
  auto swizzle_atom = composition(Swizzle<3,3,3>{}, swizzle_base);
  auto sA_layout = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
  auto sB_layout = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));

  // === G2S copies: match sgemm_sm80.cu ===
  using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
  auto copyA = make_tiled_copy(
      G2SAtom{},
      Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // 16x8 threads, k-major
      Layout<Shape<_1,_8>>{});                // 1x8 values
  auto copyB = make_tiled_copy(
      G2SAtom{},
      Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // 16x8 threads, k-major
      Layout<Shape<_1,_8>>{});

  // Optional layout reference PDFs (print_latex-based).
  if (opt.emit_layout_pdfs) {
#if !defined(_WIN32)
    fs::path layout_dir = outdir / "layouts";
    fs::create_directories(layout_dir);

    dump_stdout_to_file((build / "copyA.tex").string(), [&] { print_latex(copyA); });
    dump_stdout_to_file((build / "copyB.tex").string(), [&] { print_latex(copyB); });
    force_pdf14_in_tex(build / "copyA.tex");
    force_pdf14_in_tex(build / "copyB.tex");

    run_pdflatex(build, "copyA.tex");
    run_pdflatex(build, "copyB.tex");
    safe_copy(build / "copyA.pdf", layout_dir / "copyA.pdf");
    safe_copy(build / "copyB.pdf", layout_dir / "copyB.pdf");
    std::printf("Wrote %s\n", (layout_dir / "copyA.pdf").string().c_str());
    std::printf("Wrote %s\n", (layout_dir / "copyB.pdf").string().c_str());
#endif
  }

  // === CTA-local gmem tensors (layout algebra only) ===
  // Offset model: off = row*ld + k_tile*bK + k
  constexpr int kLd = kB_K * kKTileCount; // preset
  auto gA_shape = make_shape(bM, bK, Int<kKTileCount>{});
  auto gB_shape = make_shape(bN, bK, Int<kKTileCount>{});
  auto gA_stride = make_stride(Int<kLd>{}, Int<1>{}, bK);
  auto gB_stride = make_stride(Int<kLd>{}, Int<1>{}, bK);
  Tensor gA = make_tensor(make_gmem_ptr((T const*)nullptr), gA_shape, gA_stride);
  Tensor gB = make_tensor(make_gmem_ptr((T const*)nullptr), gB_shape, gB_stride);

  using CopyA = decltype(copyA);
  using CopyB = decltype(copyB);
  auto tidfrg_gA = CopyA::tidfrg_S(gA.layout());
  auto tidfrg_gB = CopyB::tidfrg_S(gB.layout());

  // Infer (V, cpy_m, cpy_k) from the per-thread sliced layout at k_tile=0.
  int V = 0, cpy_m = 0, cpy_k = 0, steps_per_stage = 0;
  {
    auto [tAgA_layout_all, thr_off] =
        slice_and_offset(make_coord(Int<0>{}, _, repeat<rank_v<decltype(gA)>>(_)), tidfrg_gA);
    (void)thr_off;
    auto tAgA_layout = slice(make_coord(_, _, _, Int<0>{}), tAgA_layout_all); // (V,cpy_m,cpy_k)
    V = int(size<0>(tAgA_layout));
    cpy_m = int(size<1>(tAgA_layout));
    cpy_k = int(size<2>(tAgA_layout));
    steps_per_stage = cpy_m * cpy_k;
  }
  if (V <= 0 || steps_per_stage <= 0) {
    throw std::runtime_error("Failed to infer G2S (V, steps_per_stage) from CUTE copy layout.");
  }

  fs::create_directories(outdir / "steps");
  if (!opt.emit_stage_pdf && !opt.emit_step_pdfs) {
    return; // layouts-only mode
  }

  // For each pipe (stage), compute per-step highlight points and render PDFs.
  int pipe_begin = 0;
  int pipe_end = kPipe;
  if (opt.pipe != -1) {
    if (opt.pipe < 0 || opt.pipe >= kPipe) {
      throw std::runtime_error("--pipe must be -1 or in [0.." + std::to_string(kPipe - 1) + "]");
    }
    pipe_begin = opt.pipe;
    pipe_end = opt.pipe + 1;
  }

  for (int pipe = pipe_begin; pipe < pipe_end; ++pipe) {
    std::vector<std::vector<ThreadBox>> boxesA(steps_per_stage);
    std::vector<std::vector<ThreadBox>> boxesB(steps_per_stage);
    for (int step = 0; step < steps_per_stage; ++step) {
      boxesA[step].reserve(kNumThreads);
      boxesB[step].reserve(kNumThreads);
    }

    for (int tid = 0; tid < kNumThreads; ++tid) {
      int warp = tid / 32;
      int lane = tid % 32;

      auto [tAgA_layout_all, thr_off_A] =
          slice_and_offset(make_coord(tid, _, repeat<rank_v<decltype(gA)>>(_)), tidfrg_gA);
      auto [tBgB_layout_all, thr_off_B] =
          slice_and_offset(make_coord(tid, _, repeat<rank_v<decltype(gB)>>(_)), tidfrg_gB);

      auto tAgA_layout = slice(make_coord(_, _, _, Int<0>{}), tAgA_layout_all); // (V,cpy_m,cpy_k)
      auto tBgB_layout = slice(make_coord(_, _, _, Int<0>{}), tBgB_layout_all);

      for (int step = 0; step < steps_per_stage; ++step) {
        int cm = step % cpy_m;
        int ck = step / cpy_m;

        ThreadBox boxA;
        boxA.tid = tid;
        boxA.warp = warp;
        boxA.lane = lane;
        for (int v = 0; v < V; ++v) {
          int off = int(thr_off_A + tAgA_layout(v, cm, ck));
          int m = off / kLd;
          int k_abs = off - m * kLd;
          int k = k_abs - opt.k_tile * kB_K;
          if (0 <= m && m < kB_M && 0 <= k && k < kB_K) {
            boxA.row_min = std::min(boxA.row_min, m);
            boxA.row_max = std::max(boxA.row_max, m);
            boxA.col_min = std::min(boxA.col_min, k);
            boxA.col_max = std::max(boxA.col_max, k);
          }
        }
        if (boxA.valid()) boxesA[step].push_back(boxA);

        ThreadBox boxB;
        boxB.tid = tid;
        boxB.warp = warp;
        boxB.lane = lane;
        for (int v = 0; v < V; ++v) {
          int off = int(thr_off_B + tBgB_layout(v, cm, ck));
          int n = off / kLd;
          int k_abs = off - n * kLd;
          int k = k_abs - opt.k_tile * kB_K;
          if (0 <= n && n < kB_N && 0 <= k && k < kB_K) {
            boxB.row_min = std::min(boxB.row_min, n);
            boxB.row_max = std::max(boxB.row_max, n);
            boxB.col_min = std::min(boxB.col_min, k);
            boxB.col_max = std::max(boxB.col_max, k);
          }
        }
        if (boxB.valid()) boxesB[step].push_back(boxB);
      }
    }

    auto sort_by_tid = [](std::vector<ThreadBox>& v) {
      std::sort(v.begin(), v.end(), [](ThreadBox const& a, ThreadBox const& b) { return a.tid < b.tid; });
    };
    for (int step = 0; step < steps_per_stage; ++step) {
      sort_by_tid(boxesA[step]);
      sort_by_tid(boxesB[step]);
    }

    // Backgrounds for this pipe (A and B separate)
    std::string bgA_name = "bgA_pipe" + std::to_string(pipe);
    std::string bgB_name = "bgB_pipe" + std::to_string(pipe);
    std::string bgA_tex = make_background_tex(plA, sA_layout, pipe, kElemBytes, "A", "m",
                                              opt.emit_layout_pdfs ? "copyA.pdf" : "");
    std::string bgB_tex = make_background_tex(plB, sB_layout, pipe, kElemBytes, "B", "n",
                                              opt.emit_layout_pdfs ? "copyB.pdf" : "");
    write_text_file(build / (bgA_name + ".tex"), bgA_tex);
    write_text_file(build / (bgB_name + ".tex"), bgB_tex);
    run_pdflatex(build, bgA_name + ".tex");
    run_pdflatex(build, bgB_name + ".tex");

    // Per-stage PDFs (multi-page), A and B separate
    std::string stageA_base = "g2sA_stage_pipe" + std::to_string(pipe) + "_ktile" + std::to_string(opt.k_tile);
    std::string stageB_base = "g2sB_stage_pipe" + std::to_string(pipe) + "_ktile" + std::to_string(opt.k_tile);
    std::string stage_base = "g2s_stage_pipe" + std::to_string(pipe) + "_ktile" + std::to_string(opt.k_tile); // interleaved A/B

    bool need_stage_pdf = opt.emit_stage_pdf || opt.emit_step_pdfs;
    if (need_stage_pdf) {
      std::string stageA_tex =
          make_stage_tex(plA, "A", "m", pipe, opt.k_tile, V, cpy_m, cpy_k, steps_per_stage, boxesA, bgA_name + ".pdf");
      std::string stageB_tex =
          make_stage_tex(plB, "B", "n", pipe, opt.k_tile, V, cpy_m, cpy_k, steps_per_stage, boxesB, bgB_name + ".pdf");
      write_text_file(build / (stageA_base + ".tex"), stageA_tex);
      write_text_file(build / (stageB_base + ".tex"), stageB_tex);
      run_pdflatex(build, stageA_base + ".tex");
      run_pdflatex(build, stageB_base + ".tex");

      if (opt.emit_stage_pdf) {
        safe_copy(build / (stageA_base + ".pdf"), outdir / (stageA_base + ".pdf"));
        safe_copy(build / (stageB_base + ".pdf"), outdir / (stageB_base + ".pdf"));
        std::printf("Wrote %s\n", (outdir / (stageA_base + ".pdf")).string().c_str());
        std::printf("Wrote %s\n", (outdir / (stageB_base + ".pdf")).string().c_str());

        if (opt.emit_combined_pdf) {
          std::string combined_tex = make_interleaved_stage_tex(stageA_base + ".pdf", stageB_base + ".pdf", steps_per_stage);
          write_text_file(build / (stage_base + ".tex"), combined_tex);
          run_pdflatex(build, stage_base + ".tex");
          safe_copy(build / (stage_base + ".pdf"), outdir / (stage_base + ".pdf"));
          std::printf("Wrote %s\n", (outdir / (stage_base + ".pdf")).string().c_str());
        }
      }
    }

    // Per-step PDFs: extract pages from the A/B stage PDFs using pdfpages.
    if (opt.emit_step_pdfs) {
      fs::path step_dir = outdir / "steps" / ("pipe" + std::to_string(pipe) + "_ktile" + std::to_string(opt.k_tile));
      fs::create_directories(step_dir);

      std::string extract_tex_name = "extract_step.tex";
      for (int step = 0; step < steps_per_stage; ++step) {
        auto emit_one = [&](char const* prefix, std::string const& stage_pdf) {
          std::ostringstream job;
          job << prefix << "_step_";
          job.width(3);
          job.fill('0');
          job << step;
          std::string extract_tex = make_extract_page_tex(stage_pdf, step + 1);
          write_text_file(build / extract_tex_name, extract_tex);
          run_pdflatex(build, extract_tex_name, job.str());
          safe_rename(build / (job.str() + ".pdf"), step_dir / (job.str() + ".pdf"));
        };
        emit_one("A", stageA_base + ".pdf");
        emit_one("B", stageB_base + ".pdf");
      }
      std::printf("Wrote per-step PDFs in %s\n", step_dir.string().c_str());
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    Options opt = parse_opts(argc, argv);
    generate(opt);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
