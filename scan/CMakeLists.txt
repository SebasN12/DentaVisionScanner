cmake_minimum_required(VERSION 3.20)

project(DentaVisionScanner)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(
    DentaVisionScanner

    src/main.cpp

    # Packet capture and analysis
    src/capture/GvspReceiver.cpp
    src/capture/GvspAnalyzer.cpp

    # Mock hardware
    src/hardware/mock/MockScannerHardware.cpp

    # Hardware / network communication
    src/hardware/UDPClient.cpp
    src/hardware/GvcpClient.cpp
    src/hardware/GvcpSequence.cpp

    # Scanner
    src/scanner/Scanner.cpp
)


if(WIN32)
    target_link_libraries(
        DentaVisionScanner
        ws2_32
    )
endif()