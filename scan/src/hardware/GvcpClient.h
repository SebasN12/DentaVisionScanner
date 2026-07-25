#pragma once

#include <cstdint>
#include "UDPClient.h"

class GvcpClient
{
public:

    GvcpClient(UDPClient& udp);

    bool readRegister(uint32_t address);

    bool writeRegister(uint32_t address,
                       uint32_t value);

private:

    UDPClient& m_udp;

    uint16_t m_requestId = 1;

};