#pragma once

#include "../hardware/IScannerHardware.h"


class Scanner
{

private:

    IScannerHardware* hardware;


public:

    Scanner(IScannerHardware* hw);


    void runCapture();

};