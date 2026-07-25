#include "GvcpClient.h"

#include <iostream>
#include <vector>




GvcpClient::GvcpClient(UDPClient& udp)
    :
    m_udp(udp)
{
}


bool GvcpClient::readRegister(uint32_t address)
{


    std::vector<uint8_t> packet;



    // -------------------------
    // GVCP HEADER
    // -------------------------


    // Key
    packet.push_back(0x42);


    // Flags
    packet.push_back(0x00);



    // Command = READREG (0x0080)

    packet.push_back(0x00);

    packet.push_back(0x80);



    // Payload length = 4 bytes

    packet.push_back(0x00);

    packet.push_back(0x04);



    // Request ID

    packet.push_back(
        (m_requestId >> 8) & 0xFF
    );

    packet.push_back(
        m_requestId & 0xFF
    );



    // -------------------------
    // ADDRESS
    // -------------------------


    packet.push_back(
        (address >> 24) & 0xFF
    );

    packet.push_back(
        (address >> 16) & 0xFF
    );

    packet.push_back(
        (address >> 8) & 0xFF
    );

    packet.push_back(
        address & 0xFF
    );


    m_requestId++;


    if(!m_udp.sendPacket(packet))
    {
        return false;
    }


    std::vector<uint8_t> response;



    if(!m_udp.receivePacket(response,1000))
    {

        std::cout
            << "No ACK received\n";

        return false;

    }


    std::cout
        << "GVCP response:\n";



    for(uint8_t b : response)
    {
        printf("%02X ", b);
    }


    printf("\n");



    return true;

}