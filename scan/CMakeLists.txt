cmake_minimum_required(VERSION 3.20)

project(DentaVisionScanner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(
    DentaVisionScanner

    src/main.cpp

    # Hardware
    src/hardware/mock/MockScannerHardware.cpp
    src/hardware/GvcpClient.cpp
    src/hardware/GvcpSequence.cpp
    src/capture/GvspReceiver.cpp
    src/capture/GvspAnalyzer.cpp

    # Network capture / UDP
    src/hardware/UDPClient.cpp
    src/capture/CaptureSession.cpp

    # Scanner
    src/scanner/Scanner.cpp
)


if(WIN32)
    target_link_libraries(
        DentaVisionScanner
        ws2_32
    )
endif()