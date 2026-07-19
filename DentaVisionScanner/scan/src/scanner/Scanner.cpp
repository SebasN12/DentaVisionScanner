#include <iostream>

#include "Scanner.h"


Scanner::Scanner(IScannerHardware* hw)
{
    hardware = hw;
}


void Scanner::runCapture()
{

    hardware->connect();


    for(int i=0;i<5;i++)
    {
        hardware->projectPattern(i);

        hardware->captureImage();
    }


    hardware->disconnect();

}