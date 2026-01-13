#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
    print("========== 2D logical_divide ==========\n\n");
    
    auto layout_2d = make_layout(make_shape(Int<8>{}, Int<8>{}));
    
    print("Original 2D layout: "); print(layout_2d); print("\n");
    print_layout(layout_2d);
    print("\n");
    
    auto tile_2d = make_tile(make_layout(Int<4>{}),   // M 方向每 4 个一组
                             make_layout(Int<2>{}));  // N 方向每 2 个一组
    
    auto divided_2d = logical_divide(layout_2d, tile_2d);
    
    print("2D Tile: "); print(tile_2d); print("\n");
    print("After 2D logical_divide: "); print(divided_2d); print("\n");
    print_layout(divided_2d);
    print("\n");
    
    print("========== zipped_divide ==========\n\n");
    
    // 5. 对比 zipped_divide
    auto zipped = zipped_divide(layout_2d, tile_2d);
    
    print("After zipped_divide: "); print(zipped); print("\n");
    print_layout(zipped);
    print("\n");
    
    return 0;
}
