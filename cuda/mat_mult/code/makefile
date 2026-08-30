# Adjust -arch to match your GPU: sm_75=Turing, sm_80=Ampere, sm_89=Ada
# Tensor cores require sm_70+
NVCC  = nvcc
ARCH  = -arch=sm_120
FLAGS = -Iinclude -lcublas $(ARCH)

SRCS  = src/mat_mul_base.cu src/mat_mul_cublas.cu src/mat_mul_tiled.cu src/mat_mul_regtile.cu src/mat_mul_float4.cu test.cu

all:
	$(NVCC) $(SRCS) $(FLAGS) -o test

clean:
	rm -f test
