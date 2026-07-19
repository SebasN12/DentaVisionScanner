#include <iostream>

#include "hardware/MockScannerHardware.h"
#include "scanner/Scanner.h"


int main()
{
    std::cout << "[INFO] DentaVision Scanner starting...\n";


    MockScannerHardware hardware;


    Scanner scanner(&hardware);


    scanner.runCapture();


    std::cout << "[INFO] Session finished\n";


    return 0;
}