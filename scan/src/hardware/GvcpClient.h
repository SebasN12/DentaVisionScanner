#pragma once

#include <cstdint>
#include <vector>

#include "UDPClient.h"

class GvcpClient
{
public:

    explicit GvcpClient(UDPClient& udp);

    bool readRegister(uint32_t address,
                      uint32_t& value);

    bool writeRegister(uint32_t address,
                       uint32_t value);

private:

    bool sendCommand(const std::vector<uint8_t>& packet,
                     std::vector<uint8_t>& response);

    void printPacket(const std::vector<uint8_t>& packet,
                     const char* title);

    bool parseReadAck(const std::vector<uint8_t>& response,
                      uint32_t& value);

    bool parseWriteAck(const std::vector<uint8_t>& response);

    UDPClient& m_udp;

    uint16_t m_requestId = 1;
};