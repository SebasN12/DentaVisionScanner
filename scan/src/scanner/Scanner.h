#pragma once

#include "../hardware/mock/IScannerHardware.h"


class Scanner
{

private:

    IScannerHardware* hardware;


public:

    Scanner(IScannerHardware* hw);


    void runCapture();

};