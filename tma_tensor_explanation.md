# CuTe TMA Tensor 详解

## 1. 什么是 TMA

TMA（Tensor Memory Accelerator）是 Hopper 架构引入的硬件单元，用于在全局内存和共享内存之间高效传输多维数据。

### 1.1 TMA vs 传统 Copy

| 特性 | 传统 Copy (cp.async) | TMA Copy |
|------|---------------------|----------|
| **粒度** | 每个线程加载几个元素 | 一次加载整个 tile |
| **地址计算** | 每个线程独立计算地址 | TMA 硬件自动计算 |
| **发起者** | 所有线程参与 | 通常 1 个线程发起 |
| **执行方式** | SM 执行 | 独立硬件单元异步执行 |

### 1.2 TMA 的优势

- **减少指令开销**：不需要每个线程计算地址、发起 load 指令
- **异步执行**：TMA 硬件独立于 SM 工作，SM 可以继续其他计算
- **硬件 Swizzle**：传输时自动处理 bank conflict
- **Multicast**：一次传输可以发送到多个 SM 的 shared memory

## 2. 为什么 TMA 需要 ArithTuple

### 2.1 核心问题

TMA 指令的接口如下：

```cpp
struct SM90_TMA_LOAD_3D {
  CUTE_DEVICE static void
  copy(void const* desc_ptr,      // TMA descriptor
       void const* smem_ptr,       // 目标 shared memory
       int32_t crd0,               // 坐标维度 0
       int32_t crd1,               // 坐标维度 1
       int32_t crd2) {             // 坐标维度 2
    // PTX 指令
  }
};
```

关键观察：**TMA 消费的是多维坐标 `(crd0, crd1, crd2)`，不是 1D 偏移量！**

### 2.2 普通 Layout 的问题

普通 CuTe Layout 使用整数 stride，内积结果是标量：

```cpp
// 普通 tensor
stride = (1, 128)     // 整数 stride
coord  = (2, 3)

// 内积计算
offset = 2 * 1 + 3 * 128 = 386  // 标量！维度信息丢失
```

从 `386` 无法反推出原始坐标 `(2, 3)`。

### 2.3 ArithTuple Stride 的解决方案

使用基向量作为 stride，内积结果保持为多维坐标：

```cpp
// TMA tensor
stride = (E<0>{}, E<1>{})  // 基向量 stride (1@0, 1@1)
coord  = (2, 3)

// 内积计算
result = 2 * E<0>{} + 3 * E<1>{}
       = 2 * (1,0) + 3 * (0,1)
       = (2, 0) + (0, 3)
       = (2, 3)  // 多维坐标！
```

## 3. ArithTuple 基础

### 3.1 基向量 E<i>{}

`E<i>{}` 表示第 i 维的单位向量：

| C++ 表达式 | 含义 | 打印格式 |
|-----------|------|----------|
| `E<0>{}` | `(1, 0, 0, ...)` | `1@0` |
| `E<1>{}` | `(0, 1, 0, ...)` | `1@1` |
| `E<2>{}` | `(0, 0, 1, ...)` | `1@2` |

### 3.2 运算规则

```cpp
// 缩放
3 * E<0>{} = (3, 0, 0, ...) = 3@0
5 * E<1>{} = (0, 5, 0, ...) = 5@1

// 加法
E<0>{} + E<1>{} = (1, 0) + (0, 1) = (1, 1)
3*E<0>{} + 5*E<1>{} = (3, 0) + (0, 5) = (3, 5)
```

### 3.3 类比线性代数

| 线性代数 | CuTe ArithTuple |
|---------|-----------------|
| ê₀ = (1, 0) | `E<0>{}` = `1@0` |
| ê₁ = (0, 1) | `E<1>{}` = `1@1` |
| 3ê₀ + 5ê₁ = (3, 5) | `3*E<0>{} + 5*E<1>{} = (3, 5)` |

## 4. TMA Tensor 的构造

### 4.1 基本构造

```cpp
// 创建 2D TMA 坐标 tensor
Tensor tma_tensor = make_tensor(
    make_inttuple_iter(0, 0),        // 起始坐标 (0, 0)
    make_shape(4, 5),                 // shape 4×5
    make_stride(E<0>{}, E<1>{})      // ArithTuple stride
);

// 打印格式
// ArithTuple(0,0) o (4,5):(_1@0,_1@1)
```

### 4.2 输出内容

```
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)
```

每个位置存储的是 TMA 坐标，可直接传给 TMA 指令。

### 4.3 理解打印格式

```
ArithTuple(0,0) o (4,5):(_1@0,_1@1)
     │              │       │
     │              │       └── stride: (E<0>{}, E<1>{})
     │              └── shape: 4×5
     └── 迭代器起始坐标: (0, 0)
```

- **mode 0**（行方向，size=4）stride 是 `_1@0` = `E<0>{}` = `(1, 0)`
- **mode 1**（列方向，size=5）stride 是 `_1@1` = `E<1>{}` = `(0, 1)`

### 4.4 坐标交换

通过交换 stride 可以交换坐标顺序：

```cpp
// 交换 stride
Tensor b = make_tensor(
    make_inttuple_iter(0, 0),
    make_shape(4, 5),
    make_stride(E<1>{}, E<0>{})   // 注意：交换了！
);

// 输出
// ArithTuple(0,0) o (4,5):(_1@1,_1@0):
//   (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
//   (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
//   (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
//   (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
```

## 5. 完整示例：3D Tensor Tiling

### 5.1 问题设定

- 全局 tensor: `128 × 128 × 128`
- 要 slice 出: `16 × 16 × 16` 的 tile
- 起始坐标: `(32, 32, 32)`

### 5.2 创建 TMA 坐标 Tensor

```cpp
// 1. 创建描述整个 gmem 的 TMA 坐标 tensor
Tensor tma_gmem = make_tensor(
    make_inttuple_iter(0, 0, 0),           // 起始坐标 (0,0,0)
    make_shape(128, 128, 128),              // 全局 shape
    make_stride(E<0>{}, E<1>{}, E<2>{})    // 3D 基向量
);

// 打印: ArithTuple(0,0,0) o (128,128,128):(_1@0,_1@1,_1@2)
```

### 5.3 Tiling 操作

```cpp
// 2. 划分成 16×16×16 的 tile
auto tile_shape = make_shape(Int<16>{}, Int<16>{}, Int<16>{});
Tensor tiled = zipped_divide(tma_gmem, tile_shape);

// tiled 的 shape: ((16,16,16), (8,8,8))
//                  tile内shape  tile数量(128/16=8)

// tiled 的 stride: ((_1@0,_1@1,_1@2), (_16@0,_16@1,_16@2))
//                   tile内stride      tile间stride
```

### 5.4 Slice 出特定 Tile

```cpp
// 3. 选择起始于 (32,32,32) 的 tile
//    tile 索引 = (32/16, 32/16, 32/16) = (2, 2, 2)
Tensor my_tile = tiled(_, make_coord(2, 2, 2));

// 迭代器偏移计算:
// offset = 2 * (_16@0) + 2 * (_16@1) + 2 * (_16@2)
//        = 2 * (16,0,0) + 2 * (0,16,0) + 2 * (0,0,16)
//        = (32, 0, 0) + (0, 32, 0) + (0, 0, 32)
//        = (32, 32, 32)

// my_tile 变成:
// ArithTuple(32,32,32) o (16,16,16):(_1@0,_1@1,_1@2)
```

### 5.5 访问 Tile 内的元素

```cpp
// 访问 my_tile(5, 3, 7)
//
// result = base + 5*(_1@0) + 3*(_1@1) + 7*(_1@2)
//        = (32,32,32) + 5*(1,0,0) + 3*(0,1,0) + 7*(0,0,1)
//        = (32,32,32) + (5,0,0) + (0,3,0) + (0,0,7)
//        = (37, 35, 39)

auto coord = my_tile(5, 3, 7);  // 返回 ArithTuple(37, 35, 39)
```

### 5.6 TMA 使用

```cpp
// TMA 只需要 tile 起始坐标
auto start = my_tile(0, 0, 0);  // = ArithTuple(32, 32, 32)

// 展开调用 TMA 指令
SM90_TMA_LOAD_3D::copy(
    desc_ptr,
    smem_ptr,
    get<0>(start),  // 32
    get<1>(start),  // 32
    get<2>(start)   // 32
);
// TMA 硬件自动加载从 (32,32,32) 开始的 16×16×16 数据块
```

## 6. TMA Descriptor 与坐标的关系

### 6.1 TMA Descriptor 结构

TMA descriptor 在 host 端创建，包含全局 tensor 的完整描述：

```cpp
// TMA descriptor（概念结构）
struct TmaDescriptor {
    void* gmem_base_ptr;     // 全局内存基地址
    uint64_t dim[5];         // 各维度大小
    uint64_t stride[5];      // 各维度步长（字节）
    uint32_t box_size[5];    // 每次传输的 tile 大小
    uint32_t swizzle_mode;   // 硬件 swizzle 配置
    // ... 其他配置
};
```

### 6.2 坐标如何被消费

```
TMA Descriptor (host 创建):
┌─────────────────────────────────┐
│ base_ptr = 0x7f000000           │
│ dim = {128, 128, 128}           │
│ stride = {...}                  │
│ box = {16, 16, 16}              │
└─────────────────────────────────┘
              │
              ▼
ArithTuple 坐标 (device 计算):
┌──────────────┐
│ (32, 32, 32) │  ← tile 起始坐标
└──────────────┘
              │
              ▼
TMA 硬件自动计算:
  gmem_addr = base + 32*stride[0] + 32*stride[1] + 32*stride[2]
  然后 DMA 传输 16×16×16 的数据块到 smem
```

### 6.3 各组件的职责

| 组件 | 创建位置 | 内容 | 作用 |
|------|---------|------|------|
| TMA Descriptor | Host | gmem 基地址、shape、stride、tile 大小 | 告诉硬件全局 tensor 的布局 |
| ArithTuple 坐标 | Device | 多维坐标 `(m, n, k, ...)` | 告诉硬件从哪个位置开始加载 |
| TMA 硬件 | - | - | 根据 descriptor + 坐标自动 DMA |

## 7. 实际使用流程

### 7.1 Host 端：创建 TMA Descriptor

```cpp
// 创建 TMA descriptor
auto tma_load_a = make_tma_copy(
    SM90_TMA_LOAD{},           // TMA 操作类型
    tensor_A,                   // gmem tensor（包含 layout）
    smem_layout,               // 目标 smem 布局
    tile_shape,                // 每次传输的 tile 大小
    cluster_shape              // multicast 配置
);
```

### 7.2 Device 端：计算坐标并发起 TMA

```cpp
__global__ void kernel(...) {
    // 1. 创建 TMA 坐标 tensor
    Tensor tma_coords = make_tensor(
        make_inttuple_iter(0, 0),
        make_shape(num_tiles_m, num_tiles_k),
        make_stride(E<0>{} * tile_m, E<1>{} * tile_k)
    );

    // 2. 获取当前 CTA 的 tile 坐标
    auto coord = tma_coords(cta_m_idx, cta_k_idx);

    // 3. 发起 TMA 传输
    copy(tma_load_a, coord, smem_tensor);

    // 4. 等待完成
    cp_async_wait<0>();
}
```

## 8. 复杂 Stride 示例

### 8.1 文档开头的复杂例子

```
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```

分解：

| 部分 | 内容 | 含义 |
|------|------|------|
| Iterator | `ArithTuple(0,_0,_0,_0)` | 4 维起始坐标 `(0,0,0,0)` |
| Shape | `((_128,_64),2,3,1)` | 嵌套 shape：第 0 维是 128×64，然后 2,3,1 |
| Stride | `((_1@0,_1@1),_64@1,_1@2,_1@3)` | 见下方解释 |

Stride 详解：
- `_1@0` = `E<0>{}` = `(1,0,0,0)` → 贡献到坐标第 0 维
- `_1@1` = `E<1>{}` = `(0,1,0,0)` → 贡献到坐标第 1 维
- `_64@1` = `64*E<1>{}` = `(0,64,0,0)` → 贡献到坐标第 1 维，步长 64
- `_1@2` = `E<2>{}` = `(0,0,1,0)` → 贡献到坐标第 2 维
- `_1@3` = `E<3>{}` = `(0,0,0,1)` → 贡献到坐标第 3 维

### 8.2 计算示例

访问逻辑坐标 `((i,j), k, l, m)` 其中 `i=2, j=3, k=1, l=2, m=0`：

```cpp
result = i*(_1@0) + j*(_1@1) + k*(_64@1) + l*(_1@2) + m*(_1@3)
       = 2*(1,0,0,0) + 3*(0,1,0,0) + 1*(0,64,0,0) + 2*(0,0,1,0) + 0*(0,0,0,1)
       = (2,0,0,0) + (0,3,0,0) + (0,64,0,0) + (0,0,2,0) + (0,0,0,0)
       = (2, 67, 2, 0)   // 4 维 TMA 坐标
```

## 9. 总结

### 9.1 为什么需要 ArithTuple

| 普通 Layout | TMA Layout |
|-------------|------------|
| Iterator = 指针 | Iterator = ArithTupleIterator |
| Stride = 整数 | Stride = ArithTuple（基向量）|
| Layout 输出 = 1D 偏移量 | Layout 输出 = 多维坐标 |
| 用于计算 `ptr + offset` | 用于传给 TMA `copy(..., crd0, crd1)` |

### 9.2 设计优势

1. **统一抽象**：TMA tensor 可以使用相同的 `tile`、`partition`、`slice` 操作
2. **自动追踪**：tiling/slicing 后坐标自动正确
3. **类型安全**：编译时确保坐标维度正确

### 9.3 关键记忆点

- **TMA 消费坐标，不是偏移量**
- **ArithTuple stride 保持多维坐标不被压缩**
- **坐标是 Tile 级别的，不是元素级别的**
- **TMA 硬件根据 descriptor + 坐标自动完成整个 tile 的传输**
