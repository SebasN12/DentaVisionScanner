#include <iostream>

#include "hardware/UDPClient.h"
#include "hardware/GvcpClient.h"
#include "hardware/GvcpSequence.h"


#include <iostream>

#include "hardware/GvspReceiver.h"


int main()
{

    GvspReceiver receiver(62467);


    if(!receiver.open())
    {
        return -1;
    }


    receiver.start();


    return 0;
}

// int main()
// {
//     UDPClient udp("192.168.232.2",3956);

//     if(!udp.open())
//         return -1;

//     GvcpClient gvcp(udp);

//     GvcpSequence sequence(gvcp);

//     // This two failed.
//     sequence.write(0x010E0000,0x00000000);
//     sequence.write(0x010E0000,0x00000020);


//     sequence.read(0x010E0120);
//     sequence.read(0x010E0138);
//     sequence.read(0x010E0124);
//     sequence.read(0x010E013C);
//     sequence.read(0x010E0148);

//     sequence.execute();
//     return 0;
// }

// int main()
// {
//     std::cout << "[INFO] DentaVision Scanner starting...\n";


//     MockScannerHardware hardware;


//     Scanner scanner(&hardware);


//     scanner.runCapture();


//     std::cout << "[INFO] Session finished\n";


//     return 0;
// }