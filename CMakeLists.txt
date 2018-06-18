cmake_minimum_required(VERSION 3.10)
project(Rust)

set(CMAKE_CXX_STANDARD 11)

add_executable(Rust
        rust_nn/cmake-build-debug/cmake_install.cmake
        rust_nn/cmake-build-debug/CMakeCache.txt
        rust_nn/cmake-build-debug/Makefile
        rust_nn/cmake-build-debug/rust_nn.cbp
        rust_nn/src/neural_net/layer/math.rs
        rust_nn/src/neural_net/layer/mod.rs
        rust_nn/src/neural_net/mod.rs
        rust_nn/src/lib.rs
        rust_nn/src/main.rs
        rust_nn/Cargo.lock
        rust_nn/Cargo.toml
        rust_nn/CMakeLists.txt
        README.md)
