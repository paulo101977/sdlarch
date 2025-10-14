# sdlarch

sdlarch is a small libretro frontend (sdlarch.c has less than 1000 lines of
code) created for educational purposes. It only provides the required (video,
audio and basic input) features to run basic libretro cores and there's no UI
or configuration support.

## Building

Vulkan:
```shell
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -T ClangCL
cmake --build build --config Release
```

Other than `make`, `pkg-config` and a working C99 or C++ compiler, you'll need
`sdl2` development files installed.

## Running

    ./sdlarch <core> <uncompressed content>

