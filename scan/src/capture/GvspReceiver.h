#pragma once

#include <cstdint>
#include <vector>


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


private:

    uint16_t m_port;


#ifdef _WIN32
    uintptr_t m_socket;
#else
    int m_socket;
#endif


    bool receivePacket(
        std::vector<uint8_t>& packet
    );


    uint32_t getBlockId(
        const std::vector<uint8_t>& packet
    );


    uint32_t getPacketId(
        const std::vector<uint8_t>& packet
    );


    uint16_t getPacketType(
        const std::vector<uint8_t>& packet
    );


};