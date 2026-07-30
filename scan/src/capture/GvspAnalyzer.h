#pragma once

#include <vector>
#include <cstdint>


class GvspAnalyzer
{

public:

    void analyze(
        const std::vector<uint8_t>& data
    );


private:

    uint32_t getBlockId(
        const uint8_t* packet
    );


    uint32_t getPacketId(
        const uint8_t* packet
    );

};