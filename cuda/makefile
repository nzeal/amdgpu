NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80  # adjust sm_60 to match your GPU architecture

all: dgemm_cuda

dgemm_cuda: dgemm_cuda.o
	$(NVCC) $(NVCC_FLAGS) -o dgemm_cuda dgemm_cuda.o -lcublas

dgemm_cuda.o: dgemm_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c dgemm_cuda.cu

clean:
	rm -f dgemm_cuda *.o
