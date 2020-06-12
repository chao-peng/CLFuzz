# CLFuzz: Fuzzing OpenCL Kernels

CLFuzz is a fuzzing framework for OpenCL kernels. It supports random test input generation, work-group schedule amplification and kernel execution without host code. Further detail on the design of CLFuzz can be found in the following paper:

Peng, C., & Rajan, A. (2020, February). Automated test generation for OpenCL kernels using fuzzing and constraint solving. In *Proceedings of the 13th Annual Workshop on General Purpose Processing using Graphics Processing Unit* (pp. 61-70).

```bibtex
@inproceedings{peng2020automated,
  title={Automated test generation for OpenCL kernels using fuzzing and constraint solving},
  author={Peng, Chao and Rajan, Ajitha},
  booktitle={Proceedings of the 13th Annual Workshop on General Purpose Processing using Graphics Processing Unit},
  pages={61--70},
  year={2020}
}
```

- [CLFuzz: Fuzzing OpenCL Kernels](#clfuzz-fuzzing-opencl-kernels)
  - [Prerequisite](#prerequisite)
    - [A working OpenCL driver](#a-working-opencl-driver)
    - [Python and required packages](#python-and-required-packages)
    - [LLVM with Clang](#llvm-with-clang)
    - [CMake and the Ninja build tool](#cmake-and-the-ninja-build-tool)
  - [Build the tool](#build-the-tool)
  - [Usage](#usage)
    - [Instrument the kernel code](#instrument-the-kernel-code)
    - [Test input generation](#test-input-generation)
    - [Kernel execution](#kernel-execution)

## Prerequisite

### A working OpenCL driver

Depending on your GPU vendor and operating system, the installation of OpenCL drivers varies. Most of the systems do not require manual installation and configuration. 

### Python and required packages

Python 3 and the following Python packages are required to run our script

- Numpy
- [PyOpenCL](https://documen.tician.de/pyopencl/)
- [PyYAML](https://pyyaml.org/)

Please install them with your favourite Python package manager. 

### LLVM with Clang

LLVM is evolving very fast due to its active development. Our project works fine with LLVM 8 and the following guide assumes that you want also use LLVM 8.

**Download LLVM with Clang**

Please download the following packages from the [LLVM Download Page](https://releases.llvm.org/download.html#8.0.1)

- [LLVM source code](https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/llvm-8.0.1.src.tar.xz), decompress the tarball and rename it to *llvm-src*
- [Clang source code](https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/cfe-8.0.1.src.tar.xz), decompress the tarball, rename it to *clang* and move it into *llvm-src/tools/*
- [clang-tools-extra](https://github.com/llvm/llvm-project/releases/download/llvmorg-8.0.1/clang-tools-extra-8.0.1.src.tar.xz), decompress the tarball, rename it to *extra* and move it into *llvm-src/tools/clang/tools/*

**Plug our tools into LLVM**

Place the following directories into *llvm-src/tools/clang/tools/extra/* directory:

- clscheduler, as found in this repository
- clfuzzer, as found in this repository

Open the *CMakeLists.txt* file in *llvm-src/tools/clang/tools/extra/* and add the following lines:

```CMake
add_subdirectory(clfuzzer)
add_subdirectory(clschedule)
```

### CMake and the Ninja build tool

[CMake](https://cmake.org/) is officially used by LLVM and Clang to generate build scripts and [Ninja](https://ninja-build.org/) is recommended by LLVM to build it.

If you use Debian/Ubuntu, simple run the following command to install Ninja:

```bash
apt-get install ninja-build
```

## Build the tool

Create *llvm-build* and *llvm-install* directories in the same folder where *llvm-src" resides so that all these three directories are in the same folder:

- llvm-build
- llvm-install
- llvm-src

Open your terminal, go to *llvm-build* and issue the following commands (remember to give a correct absolute path to *llvm-install*):

```bash
cmake -G Ninja ../llvm-src -DLLVM_BUILD_TESTS=ON -CMAKE_BUILD_TYPE=MinSizeRel CMAKE_INSTALL_PREFIX=absolute-path-to-llvm-install 
```

The command takes around a minute to generate CMake files, once it finishes, build the tools using:

```bash
ninja
ninja install
```

It takes around 20 minutes to build everything and the second command moves all the executables and libraries to the *llvm-install* folder.

## Usage

Before running any command of CLFuzz, make sure the following paths are in your PATH environment variable so that the system can find all the commands:

- Path to *llvm-install/bin*
- Path to *clexec*

### Instrument the kernel code

Commands *clfuzzer* and *clschedule* built with the LLVM framework are used to instrument the OpenCL kernel code.

```bash
mkdir clfuzz-output
clfuzzer kernel.cl
clschedule kernel.cl
```

These two commands generate instrumented OpenCL kernel source code files (kernel_cov.cl and kernel_schedule.cl) and an YAML file (kernel.cl.yaml) describing the interface of the kernel which are used for our script to generate test inputs and execute them.

### Test input generation

```bash
cltestgen kernel.cl.yaml -n NUMBER_OF_TESTS
```

This command generates specified number of tests in npy format.

### Kernel execution

If you want to run the kernel without coverage measurement:

```bash
clexec kernel.cl -n NUMBER_OF_TESTS
```

If you want to run the kernel with coverage measurement, replace the original kernel.cl file with kernel_cov.cl and:

```bash
clexec kernel.cl -n NUMBER_OF_TESTS -cov
```

If you want to enable work-group schedule,

```bash
clschedulegen -info kernel.cl.yaml -g GLOBAL_SIZE -l LOCAL_SIZE -n NUMBER_OF_SCHEDULES
clexec kernel_schedule.cl -schedule
```

