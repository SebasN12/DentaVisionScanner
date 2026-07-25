#include <iostream>

#include "hardware/UDPClient.h"
#include "hardware/GvcpClient.h"


int main()
{

    UDPClient udp(
        "192.168.232.2",
        3956,
        58137
    );


    if(!udp.open())
    {
        return -1;
    }


    GvcpClient gvcp(udp);



    std::cout 
        << "Sending READREG CCP...\n";



    gvcp.readRegister(0x00000A00);



    return 0;
}

// int main()
// {
//     std::cout << "[INFO] DentaVision Scanner starting...\n";


//     MockScannerHardware hardware;


//     Scanner scanner(&hardware);


//     scanner.runCapture();


//     std::cout << "[INFO] Session finished\n";


//     return 0;
// }