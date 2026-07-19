cmake_minimum_required(VERSION 3.20)

project(DentaVisionScanner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(
    DentaVisionScanner

    src/main.cpp

    src/hardware/MockScannerHardware.cpp

    src/capture/CaptureSession.cpp

    src/scanner/Scanner.cpp
)