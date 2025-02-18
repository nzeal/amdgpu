# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -I./includes -arch=native
LDFLAGS = -lcudart

# Source files
SRCS = src/dgemm_cuda.cu src/matrix_mul_kernel1.cu src/matrix_mul_kernel2.cu
OBJS = $(SRCS:.cu=.o)

# Output executable
TARGET = dgemm_cuda

# Default target
all: $(TARGET)

# Build target executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LDFLAGS)

# Compile CUDA source files to object files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
