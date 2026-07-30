#include "GvspAnalyzer.h"

#include <iostream>
#include <map>


uint32_t GvspAnalyzer::getBlockId(
    const uint8_t* packet
)
{
    return
        (packet[2] << 8)
        |
        packet[3];
}



uint32_t GvspAnalyzer::getPacketId(
    const uint8_t* packet
)
{
    return
        (packet[4] << 24)
        |
        (packet[5] << 16)
        |
        (packet[6] << 8)
        |
        packet[7];
}



void GvspAnalyzer::analyze(
    const std::vector<uint8_t>& data
)
{

    const size_t packetSize = 7992;


    size_t packets =
        data.size()/packetSize;


    std::cout
        <<"Packets detected: "
        <<packets
        <<"\n";


    std::map<uint32_t,int> blocks;



    for(size_t i=0;i<packets;i++)
    {

        const uint8_t* packet =
            &data[i*packetSize];


        uint32_t block =
            getBlockId(packet);


        uint32_t id =
            getPacketId(packet);



        blocks[block]++;


        if(i<10)
        {
            std::cout
            <<"Block "
            <<block
            <<" Packet "
            <<id
            <<"\n";
        }

    }



    std::cout<<"\nFrames detected:\n";


    for(auto& b:blocks)
    {

        std::cout
        <<"Block "
        <<b.first
        <<" packets "
        <<b.second
        <<"\n";

    }


}