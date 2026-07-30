#include <iostream>

#include "hardware/UDPClient.h"
#include "hardware/GvcpClient.h"
#include "hardware/GvcpSequence.h"


#include <iostream>

#include "capture/GvspReceiver.h"

// streaming images
#include <iostream>
#include <fstream>
#include "capture/GvspAnalyzer.h"

#include <vector>
#include <filesystem>

#include "capture/GvspReceiver.h"


int main()
{

    std::cout 
        << "Working directory: "
        << std::filesystem::current_path()
        << std::endl;


    uint16_t gvspPort = 62467;


    GvspReceiver receiver(gvspPort);


    if(!receiver.open())
    {
        std::cout
            << "Could not open GVSP receiver\n";

        return -1;
    }



    std::ofstream file(
        "capture.raw",
        std::ios::binary);



    if(!file.is_open())
    {
        std::cout
            << "Could not create file\n";

        return -1;
    }



    std::cout
        << "Capturing packets...\n";



    size_t totalBytes = 0;


    for(int i=0;i<500;i++)
    {

        std::vector<uint8_t> packet;


        if(receiver.receivePacket(packet))
        {

            file.write(
                reinterpret_cast<char*>(packet.data()),
                packet.size()
            );


            totalBytes += packet.size();



            std::cout
                << "Packet "
                << i
                << " size "
                << packet.size()
                << "\n";

        }

    }



    file.close();

    std::cout
        << "Saved "
        << totalBytes
        << " bytes\n";


    std::ifstream input(
        "capture.raw",
        std::ios::binary
    );


    std::vector<uint8_t> data(
        std::istreambuf_iterator<char>(input),
        {}
    );


    GvspAnalyzer analyzer;

    analyzer.analyze(data);

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