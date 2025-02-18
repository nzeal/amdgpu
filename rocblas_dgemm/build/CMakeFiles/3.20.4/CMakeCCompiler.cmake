set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.23/bin/cc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "16.0.6")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "11")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "")




set(CMAKE_AR "/opt/cray/pe/cce/16.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ar")
set(CMAKE_C_COMPILER_AR "CMAKE_C_COMPILER_AR-NOTFOUND")
set(CMAKE_RANLIB "/opt/cray/pe/cce/16.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "CMAKE_C_COMPILER_RANLIB-NOTFOUND")
set(CMAKE_LINKER "/opt/cray/pe/cce/16.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "x86_64-unknown-linux-gnu")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-unknown-linux-gnu")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/cray/pe/libsci/23.09.1.1/CRAY/12.0/x86_64/include;/opt/cray/pe/mpich/8.1.27/ofi/cray/14.0/include;/opt/rocm-6.0.3/include;/opt/rocm-6.0.3/include/rocprofiler;/opt/rocm-6.0.3/include/roctracer;/opt/cray/xpmem/2.8.2-1.0_5.1__g84a27a5.shasta/include;/opt/cray/pe/dsmml/0.2.2/dsmml/include;/opt/cray/pe/cce/16.0.1/cce-clang/x86_64/lib/clang/16/include;/opt/cray/pe/cce/16.0.1/cce/x86_64/include/craylibs;/usr/local/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "amdhip64;sci_cray_mpi;sci_cray;dl;mpi_cray;mpi_gtl_hsa;dsmml;xpmem;stdc++;pgas-shmem;quadmath;modules;fi;craymath;f;u;csup;pthread;atomic;m;unwind;c;unwind")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/libsci/23.09.1.1/CRAY/12.0/x86_64/lib;/opt/cray/pe/mpich/8.1.27/ofi/cray/14.0/lib;/opt/cray/pe/mpich/8.1.27/gtl/lib;/opt/rocm-6.0.3/lib;/opt/rocm-6.0.3/lib/rocprofiler;/opt/rocm-6.0.3/lib/roctracer;/opt/cray/xpmem/2.8.2-1.0_5.1__g84a27a5.shasta/lib64;/opt/cray/pe/dsmml/0.2.2/dsmml/lib;/opt/cray/pe/cce/16.0.1/cce/x86_64/lib;/opt/cray/pe/cce/16.0.1/cce-clang/x86_64/lib/x86_64-unknown-linux-gnu;/opt/cray/pe/gcc/10.3.0/snos/lib/gcc/x86_64-suse-linux/10.3.0;/opt/cray/pe/gcc/10.3.0/snos/lib64;/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib64;/lib;/usr/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
