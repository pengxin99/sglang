icpx -pthread -B /home/gta/miniforge3/envs/sglang_py310/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/gta/miniforge3/envs/sglang_py310/include -fPIC -O2 -isystem /home/gta/miniforge3/envs/sglang_py310/include -fPIC -I/home/gta/pengxin/sglang/sgl-kernel/include -I/home/gta/pengxin/sglang/sgl-kernel/csrc -I/home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/include -I/home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/gta/miniforge3/envs/sglang_py310/include/python3.10 -c -fsycl -fsycl-targets=spir64_gen,spir64 -std=c++17 -sycl-std=2020 -c -x c++ /home/gta/pengxin/sglang/sgl-kernel/csrc/xpu/uni_esimd_kernel.cpp -o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/uni_esimd_kernel_1.o -fsycl -ffast-math -fsycl-device-code-split=per_kernel -fsycl-targets=spir64_gen -DBUILD_ESIMD_KERNEL_LIB -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1016"' -DTORCH_EXTENSION_NAME=common_ops -D_GLIBCXX_USE_CXX11_ABI=1
ll /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/

cp /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/uni_esimd_kernel_1.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/uni_esimd_kernel.o
ll /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/

icpx /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/awq_dequantize.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/torch_extension_sycl.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/uni_esimd_kernel.o -o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/sycl_dlink.o -fsycl -fsycl-targets=spir64_gen,spir64 -fsycl-link --offload-compress -Xs "-device bmg-g21"


export ORIGIN
ll /home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/
ls /home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/
ls /home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/lib

cd ../..
ls build
g++ -pthread -B /home/gta/miniforge3/envs/sglang_py310/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/gta/miniforge3/envs/sglang_py310/include -fPIC -O2 -isystem /home/gta/miniforge3/envs/sglang_py310/include -shared /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/awq_dequantize.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/torch_extension_sycl.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/uni_esimd_kernel.o /home/gta/pengxin/sglang/sgl-kernel/build/temp.linux-x86_64-cpython-310/csrc/xpu/sycl_dlink.o -L/home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_python -ltorch_xpu -o build/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so -Wl,-rpath,/home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/torch/lib -L/usr/lib/x86_64-linux-gnu
ll build/lib.linux-x86_64-cpython-310/sgl_kernel/ -th
date


sudo find -name "common_ops.cpython-310-x86_64-linux-gnu.so*" ~/
sudo find ~/ -name "common_ops.cpython-310-x86_64-linux-gnu.so*"
ll /home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/sgl_kernel/
ll /home/gta/pengxin/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-310/sgl_kernel/
cp /home/gta/pengxin/sglang/sgl-kernel/build/lib.linux-x86_64-cpython-310/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so /home/gta/miniforge3/envs/sglang_py310/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so