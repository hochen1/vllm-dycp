# vLLM-DYCP
+ support Dynamic Context parellem on Decoders
## install vLLM
``` shell
VLLM_TARGET_DEVICE=empty pip install -U -e . -i https://pypi.antfin-inc.com/simple/
```
## install vLLM-Ascend
```shell
# torch_npu ：torch_npu-2.8.0.post1-cp311-cp311-manylinux_2_28_aarch64.whl
COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation --no-deps -i https://pypi.antfin-inc.com/simple/
```
