#pragma once

#include <cstdint>
#include <vector>
#include <string>


class GvspReceiver
{

public:

    GvspReceiver(uint16_t port);

    ~GvspReceiver();


    bool open();

    void close();


    bool receiveFrame(
        std::vector<uint8_t>& image
    );

    bool receivePacket(
        std::vector<uint8_t>& packet
    );


private:

    uint16_t m_port;


#ifdef _WIN32
    uintptr_t m_socket;
#else
    int m_socket;
#endif


    uint32_t getBlockId(
        const std::vector<uint8_t>& packet
    );


    uint32_t getPacketId(
        const std::vector<uint8_t>& packet
    );


};