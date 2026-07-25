#pragma once

#include <string>
#include <vector>
#include <cstdint>

#ifdef _WIN32
#include <winsock2.h>
#endif


class UDPClient
{
public:

    UDPClient(const std::string& remoteIp,
              uint16_t remotePort,
              uint16_t localPort = 0);

    ~UDPClient();


    bool open();

    void close();


    bool sendPacket(const std::vector<uint8_t>& data);


    bool receivePacket(std::vector<uint8_t>& data,
                       int timeoutMs = 1000);


private:

    std::string m_remoteIp;

    uint16_t m_remotePort;
    uint16_t m_localPort;


#ifdef _WIN32
    SOCKET m_socket;
#else
    int m_socket;
#endif

};