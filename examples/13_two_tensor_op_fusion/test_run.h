/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#include <iostream>  // C++标准输入输出流

// Run tests on GPUs  // 在GPU上运行测试

// 测试运行函数：检查GPU架构并运行测试函数
int testRun(int arch,                                          // 目标架构（如 70 = SM70, 75 = SM75, 80 = SM80）
            std::vector<bool (*)()> & test_funcs,              // 测试函数指针数组
            const std::string & test_name) {                   // 测试名称

  bool supported = false;  // 是否支持当前架构的标志

  int arch_major = arch / 10;               // 提取主版本号（如 SM80 -> 8）
  int arch_minor = arch - arch / 10 * 10;   // 提取次版本号（如 SM75 -> 5）  

  if(arch_major >= 8) {  // Ampere架构（SM80及以上）
    // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
    // Ampere Tensor Core操作通过mma.sync指令在CUDA 11.0首次可用
    //
    // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
    // CUTLASS必须使用CUDA 11工具包编译才能运行Conv2dFprop示例
    if (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0)) {
      supported = true;  // CUDA版本满足要求
    }
  }
  else if(arch_major >= 7) {  // Turing架构（SM70, SM75）
    // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
    // Turing Tensor Core操作通过mma.sync指令在CUDA 10.2首次可用
    //
    // CUTLASS must be compiled with CUDA 10.2 Toolkit to run these examples.
    // CUTLASS必须使用CUDA 10.2工具包编译才能运行这些示例
    if (__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)) {
      supported = true;  // CUDA版本满足要求
    }
  }

  cudaDeviceProp props;  // CUDA设备属性结构体

  // 获取设备0的属性
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;  // 获取设备属性失败
  }

  // 检查实际GPU架构是否与目标架构匹配
  if (!(props.major == arch_major && props.minor == arch_minor)) {
    supported = false;  // 架构不匹配
  }

  if (!supported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    // 返回0以便在旧版本工具包上测试通过。其操作为空操作。
    std::cout << "This example isn't supported on current architecture" << std::endl;  // 当前架构不支持此示例
    return 0;
  }

  bool pass = true;  // 测试通过标志

  // 输出设备信息
  std::cout << "Device: " << props.name << std::endl;         // 设备名称
  std::cout << "Arch: SM" << arch << std::endl;               // 架构版本
  std::cout << "Test: " << test_name << std::endl;            // 测试名称

  // 运行所有测试函数
  for(auto func : test_funcs) {
    pass &= func();  // 运行测试并更新通过状态
  }


  if(pass)
    return 0;   // 所有测试通过
  else
    return -1;  // 有测试失败

}

