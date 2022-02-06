# Butter
---

A library implementing utility functions from Super Mario 64's code.

Currently contains:
- SM64 trigonometric functions (`sins`, `coss`, `atan2s`, `atan2f`)
- SM64's RNG
- Linear algebra for small vectors and 2D matrices

I have yet to add a test battery and many other useful functions.

Because Tyler Kehne's [scripting framework](https://github.com/TylerKehne/sm64-tas-scripting) uses C++20 features, this library is also written for C++20. However, I use only a couple here:
- Concepts and constraints
- Template lambdas

These features should be supported well in GCC, Clang, and MSVC++.

# Building
This project can be built just like any other CMake project.
```sh
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

I plan to add support for Conan later.