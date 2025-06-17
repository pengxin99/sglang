# SGL Kernel issue repro

## deps
torch==2.8
torch-triton-xpu==3.3.1
oneapi==2025.1
pytest

## repo
```bash 
git clone -b xpu_build_back_issue_repro https://github.com/pengxin99/sglang.git
cd sglang/sglang/kernel
## install kernel
## use the torch BuildExtension in setup_sycl.py 
## from torch.utils.cpp_extension import BuildExtension, SyclExtension
python setup_sycl.py install
python tests/test_awq_dequant.py



## with torch==2.8
FAILED tests/test_awq_dequant.py::test_awq_dequant_compare_implementations[1024-128-False] - RuntimeError: No kernel named _ZTS26AWQDequantizeKernelFunctorIN4sycl3_V16detail9half_impl4halfES4_Li16ELi16ELi16EE was found

## with torch==2.7
pass
```

