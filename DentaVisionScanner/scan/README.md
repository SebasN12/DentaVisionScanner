## Build

Requirements:
- Visual Studio Build Tools with C++
- CMake

### Debug Build:

build: 

```bash
cmake -S . -B build
cmake --build build
```

Run:

```bash
build/Debug/DentaVisionScanner.exe
```

### Release Build

build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run:
```bash
build/Release/DentaVisionScanner.exe
```