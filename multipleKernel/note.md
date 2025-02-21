# Kernel 1
perf stat -e cycles,instructions,cache-references,cache-misses ./dgemm_cuda 1 1024

# Kernel 2
perf stat -e cycles,instructions,cache-references,cache-misses ./dgemm_cuda 2 1024

# Kernel 3
perf stat -e cycles,instructions,cache-references,cache-misses ./dgemm_cuda 3 1024

# Kernel 4
perf stat -e cycles,instructions,cache-references,cache-misses ./dgemm_cuda 4 1024

# Kernel 5
perf stat -e cycles,instructions,cache-references,cache-misses ./dgemm_cuda 5 1024
