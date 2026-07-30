#include <iostream>

#include "hardware/UDPClient.h"
#include "hardware/GvcpClient.h"
#include "hardware/GvcpSequence.h"


#include <iostream>

#include "capture/GvspReceiver.h"

// streaming images
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

#include "capture/GvspReceiver.h"


int main()
{

    std::cout
        << "Working directory: "
        << std::filesystem::current_path()
        << std::endl;



    /*
        GVSP port. Change manually if Sirona uses another.
    */

    uint16_t gvspPort = 62467;



    GvspReceiver receiver(gvspPort);



    if(!receiver.open())
    {
        std::cout
            << "Could not open GVSP receiver\n";

        return -1;
    }



    std::cout
        << "Waiting for frame...\n";



    std::vector<uint8_t> image;



    if(!receiver.receiveFrame(image))
    {
        std::cout
            << "Could not receive frame\n";

        return -1;
    }





    std::cout
        << "Frame received\n"
        << "Size: "
        << image.size()
        << " bytes\n";





    std::ofstream file(
        "frame.raw",
        std::ios::binary
    );



    if(!file.is_open())
    {
        std::cout
            << "Could not create frame.raw\n";

        return -1;
    }



    file.write(
        reinterpret_cast<char*>(image.data()),
        image.size()
    );


    file.close();



    std::cout
        << "Saved frame.raw\n";



    receiver.close();



    return 0;
}


// ------------------------
// SENDING GVCP COMMANDS
// ------------------------

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


// ------------------------
// Mock scanning hardware
// ------------------------

// int main()
// {
//     std::cout << "[INFO] DentaVision Scanner starting...\n";


//     MockScannerHardware hardware;


//     Scanner scanner(&hardware);


//     scanner.runCapture();


//     std::cout << "[INFO] Session finished\n";


//     return 0;
// }