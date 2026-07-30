#pragma once

#include <cstdint>
#include <string>
#include <vector>


class GvspReceiver
{

public:

    GvspReceiver(uint16_t port);

    ~GvspReceiver();


    bool open();

    void close();


    void start();


private:

    uint16_t m_port;


#ifdef _WIN32

    uintptr_t m_socket;

#else

    int m_socket;

#endif


    void processPacket(
        const std::vector<uint8_t>& packet,
        const std::string& senderIp,
        uint16_t senderPort
    );


    void printHex(
        const std::vector<uint8_t>& data
    );


    void analyzeGVSP(
        const std::vector<uint8_t>& packet
    );

};