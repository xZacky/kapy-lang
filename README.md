# kapy-lang

A customized version of the [triton-lang](https://github.com/triton-lang/triton) using affine map to describe the tensor and memory layout.

## Build From Source

### Install LLVM + MLIR From Source

```
$ git clone https:://github.com/llvm/llvm-project
$ cd llvm-project
$ git checkout llvmorg-18.1.8

$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTION=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TEST=ON \

$ ninja
$ sudo ninja install
```

### Build kapy-lang

```
$ cd kapy-lang
$ mkdir build && cd build
$ cmake -G Ninja ..
$ ninja
```

## Running Tests

```
$ cd kapy-lang/build
$ pip install lit
$ lit test
```
