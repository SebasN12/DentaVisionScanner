#include <iostream>

#include "MockScannerHardware.h"


bool MockScannerHardware::connect()
{
    std::cout << "Camera connected\n";
    return true;
}


bool MockScannerHardware::disconnect()
{
    std::cout << "Camera disconnected\n";
    return true;
}


bool MockScannerHardware::projectPattern(int pattern)
{
    std::cout 
        << "Projecting pattern "
        << pattern
        << "\n";

    return true;
}


bool MockScannerHardware::captureImage()
{
    std::cout << "Capturing image\n";

    return true;
}