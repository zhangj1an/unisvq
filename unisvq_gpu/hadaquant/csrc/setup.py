from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="HadaQuant",
    ext_modules=[
        CUDAExtension(
            name="HadaQuant",
            sources=["v1_cuda.cpp", "v1_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",   # A100/A800
                    "-gencode=arch=compute_86,code=sm_86",   # RTX 3090
                    "-gencode=arch=compute_89,code=sm_89",   # RTX 4090
                    "-gencode=arch=compute_90,code=sm_90",   # H100
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
