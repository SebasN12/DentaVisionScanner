#include "GvcpClient.h"

#include <iomanip>
#include <iostream>

GvcpClient::GvcpClient(UDPClient& udp)
    : m_udp(udp)
{
}

bool GvcpClient::readRegister(uint32_t address,
                              uint32_t& value)
{
    std::vector<uint8_t> packet;

    // Header
    packet.push_back(0x42);
    packet.push_back(0x01);

    // READREG_CMD
    packet.push_back(0x00);
    packet.push_back(0x80);

    // Payload length = 4
    packet.push_back(0x00);
    packet.push_back(0x04);

    uint16_t requestId = m_requestId++;

    packet.push_back((requestId >> 8) & 0xFF);
    packet.push_back(requestId & 0xFF);

    // Address
    packet.push_back((address >> 24) & 0xFF);
    packet.push_back((address >> 16) & 0xFF);
    packet.push_back((address >> 8) & 0xFF);
    packet.push_back(address & 0xFF);

    std::vector<uint8_t> response;

    if(!sendCommand(packet,response))
        return false;

    return parseReadAck(response,value);
}

bool GvcpClient::writeRegister(uint32_t address,
                               uint32_t value)
{
    std::vector<uint8_t> packet;

    packet.push_back(0x42);
    packet.push_back(0x01);

    // WRITEREG_CMD
    packet.push_back(0x00);
    packet.push_back(0x82);

    // Payload = address + value
    packet.push_back(0x00);
    packet.push_back(0x08);

    uint16_t requestId = m_requestId++;

    packet.push_back((requestId >> 8) & 0xFF);
    packet.push_back(requestId & 0xFF);

    packet.push_back((address >> 24) & 0xFF);
    packet.push_back((address >> 16) & 0xFF);
    packet.push_back((address >> 8) & 0xFF);
    packet.push_back(address & 0xFF);

    packet.push_back((value >> 24) & 0xFF);
    packet.push_back((value >> 16) & 0xFF);
    packet.push_back((value >> 8) & 0xFF);
    packet.push_back(value & 0xFF);

    std::vector<uint8_t> response;

    if(!sendCommand(packet,response))
        return false;

    return parseWriteAck(response);
}

bool GvcpClient::sendCommand(const std::vector<uint8_t>& packet,
                             std::vector<uint8_t>& response)
{
    printPacket(packet,"TX");

    if(!m_udp.sendPacket(packet))
        return false;

    if(!m_udp.receivePacket(response,1000))
    {
        std::cout << "Timeout waiting for ACK\n";
        return false;
    }

    printPacket(response,"RX");

    return true;
}

void GvcpClient::printPacket(const std::vector<uint8_t>& packet,
                             const char* title)
{
    std::cout
        << "\n"
        << title
        << " (" << packet.size() << " bytes)\n";

    for(size_t i=0;i<packet.size();i++)
    {
        printf("%02X ",packet[i]);

        if((i+1)%16==0)
            printf("\n");
    }

    printf("\n");
}

bool GvcpClient::parseReadAck(const std::vector<uint8_t>& response,
                              uint32_t& value)
{
    if(response.size() < 16)
    {
        std::cout << "ACK too short\n";
        return false;
    }

    uint16_t command =
        (response[2] << 8) |
        response[3];

    if(command != 0x0081)
    {
        std::cout
            << "Unexpected ACK: 0x"
            << std::hex
            << command
            << std::dec
            << "\n";

        return false;
    }

    value =
        (uint32_t(response[12]) << 24) |
        (uint32_t(response[13]) << 16) |
        (uint32_t(response[14]) << 8) |
        uint32_t(response[15]);

    std::cout
        << "Register value = 0x"
        << std::hex
        << std::uppercase
        << value
        << std::dec
        << "\n";

    return true;
}

bool GvcpClient::parseWriteAck(const std::vector<uint8_t>& response)
{
    if(response.size() < 8)
    {
        std::cout << "ACK too short\n";
        return false;
    }

    uint16_t command =
        (response[2] << 8) |
        response[3];

    if(command != 0x0083)
    {
        std::cout
            << "Unexpected ACK: 0x"
            << std::hex
            << command
            << std::dec
            << "\n";

        return false;
    }

    std::cout << "WRITE ACK received\n";

    return true;
}