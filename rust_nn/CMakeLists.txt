cmake_minimum_required(VERSION 3.10)
project(rust_nn)

set(CMAKE_CXX_STANDARD 11)

add_executable(rust_nn
        cmake-build-debug/cmake_install.cmake
        cmake-build-debug/CMakeCache.txt
        cmake-build-debug/Makefile
        cmake-build-debug/rust_nn.cbp
        src/neural_net/layer/math.rs
        src/neural_net/layer/mod.rs
        src/neural_net/mod.rs
        src/main.rs
        Cargo.lock
        Cargo.toml
        CMakeLists.txt)
