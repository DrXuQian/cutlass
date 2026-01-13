#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== logical_product 实际例子 ===\n\n";

    // 使用非连续的 TiledLayout_TV
    auto TiledLayout_TV = make_layout(make_shape(4, 4), make_stride(1, 4));
    std::cout << "TiledLayout_TV: " << TiledLayout_TV << "\n";
    print_layout(TiledLayout_TV);
    std::cout << "\n";

    // right_inverse
    auto inv_tv = right_inverse(TiledLayout_TV);
    std::cout << "right_inverse(TiledLayout_TV): " << inv_tv << "\n\n";

    // upcast<8>
    auto frg_layout_mn = upcast<8>(inv_tv.with_shape(make_shape(4, 4)));
    std::cout << "frg_layout_mn = upcast<8>(...): " << frg_layout_mn << "\n";
    print_layout(frg_layout_mn);
    std::cout << "\n";

    // right_inverse(frg_layout_mn)
    auto inv_frg = right_inverse(frg_layout_mn);
    std::cout << "right_inverse(frg_layout_mn): " << inv_frg << "\n\n";

    // logical_product
    auto V_layout = make_layout(Int<2>{});
    std::cout << "make_layout(V=2): " << V_layout << "\n\n";

    auto result = logical_product(V_layout, inv_frg);
    std::cout << "logical_product(V, right_inverse(frg_layout_mn)): " << result << "\n";

    return 0;
}
