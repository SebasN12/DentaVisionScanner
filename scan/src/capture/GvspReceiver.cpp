#include "GvspReceiver.h"

#include <iostream>

#include <fstream>

#ifdef _WIN32

#define NOMINMAX

#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib,"Ws2_32.lib")

#endif



GvspReceiver::GvspReceiver(uint16_t port)
    :
    m_port(port),
    m_socket(0)
{

}



GvspReceiver::~GvspReceiver()
{
    close();
}





bool GvspReceiver::open()
{

#ifdef _WIN32


    WSADATA data;


    if(WSAStartup(
        MAKEWORD(2,2),
        &data)!=0)
    {
        return false;
    }



    m_socket =
        socket(
            AF_INET,
            SOCK_DGRAM,
            IPPROTO_UDP);



    if(m_socket==INVALID_SOCKET)
    {
        return false;
    }



    sockaddr_in addr{};


    addr.sin_family = AF_INET;

    addr.sin_addr.s_addr = INADDR_ANY;

    addr.sin_port = htons(m_port);



    if(bind(
        m_socket,
        (sockaddr*)&addr,
        sizeof(addr)) < 0)
    {

        std::cout
            <<"Bind failed\n";

        return false;
    }



    std::cout
        <<"GVSP listening on "
        <<m_port
        <<"\n";


    return true;


#else

    return false;

#endif

}







void GvspReceiver::close()
{

#ifdef _WIN32


    if(m_socket)
    {
        closesocket(m_socket);
        m_socket=0;
    }


    WSACleanup();


#endif

}








bool GvspReceiver::receivePacket(
    std::vector<uint8_t>& packet)
{

#ifdef _WIN32


    packet.resize(9000);


    sockaddr_in sender{};


    int senderSize =
        sizeof(sender);



    int received =
        recvfrom(
            m_socket,
            (char*)packet.data(),
            packet.size(),
            0,
            (sockaddr*)&sender,
            &senderSize);



    if(received<=0)
        return false;



    packet.resize(received);


    return true;



#else

    return false;

#endif

}




uint32_t GvspReceiver::getBlockId(
    const std::vector<uint8_t>& packet)
{
    if(packet.size() < 6)
        return 0;


    return
        ((uint32_t)packet[2] << 8) |
        packet[3];
}









uint32_t GvspReceiver::getPacketId(
    const std::vector<uint8_t>& packet)
{
    if(packet.size()<8)
        return 0;


    return
        ((uint32_t)packet[6] << 16) |
        ((uint32_t)packet[7] << 8) |
        ((uint32_t)packet[8]);

}








uint16_t GvspReceiver::getPacketType(
    const std::vector<uint8_t>& packet)
{

    if(packet.size()<2)
        return 0;


    return
        ((uint16_t)packet[0]<<8)
        |
        packet[1];

}









bool GvspReceiver::receiveFrame(
    std::vector<uint8_t>& image)
{


    image.clear();


    uint32_t currentBlock = 0;


    std::cout
        <<"Waiting GVSP frame...\n";



    while(true)
    {


        std::vector<uint8_t> packet;


        if(!receivePacket(packet))
            continue;

        std::cout << "\nFirst 64 bytes:\n";

        size_t bytesToPrint = packet.size();

        if(bytesToPrint > 64)
        {
            bytesToPrint = 64;
        }

        for(size_t i = 0; i < bytesToPrint; i++)
        {
            if(i % 16 == 0)
            {
                printf("%04zX: ", i);
            }

            printf("%02X ", packet[i]);

            if((i + 1) % 16 == 0)
            {
                printf("\n");
            }
        }

        if(bytesToPrint % 16 != 0)
        {
            printf("\n");
        }

        std::cout
            << "Packet size: "
            << packet.size()
            << "\n";

        printf("Header bytes: ");

        for(size_t i = 0; i < 12 && i < packet.size(); i++)
        {
            printf("%02X ", packet[i]);
        }

        printf("\n");

        uint32_t block =
            getBlockId(packet);

        uint32_t packetId =
            getPacketId(packet);


        // if(!receivePacket(packet))
        //     continue;



        // uint32_t block =
        //     getBlockId(packet);



        // uint32_t packetId =
        //     getPacketId(packet);



        uint16_t type =
            getPacketType(packet);




        std::cout
            <<"Block: "
            <<block
            <<" Packet: "
            <<packetId
            <<" Type: 0x"
            <<std::hex
            <<type
            <<std::dec
            <<" Size: "
            <<packet.size()
            <<"\n";





        if(currentBlock==0)
        {
            currentBlock = block;
        }

        if(block != currentBlock)
        {
            std::cout
                <<"New frame detected\n";

            break;
        }


        // Leader packet
        if(packet[4] == 0x01)
        {
            std::cout
                <<"Ignoring leader packet\n";

            continue;
        }


        // Trailer packet
        if(packet[4] == 0x02)
        {
            std::cout
                <<"Trailer received\n";

            break;
        }


        // Data packets
        if(packet[4] == 0x03)
        {

            for(size_t i=8;i<packet.size();i++)
            {
                image.push_back(packet[i]);
            }

            if(image.size() > 400000)
                {
                    std::cout<<"Enough data\n";
                    break;
                }

        }


    }





    std::cout
        <<"Frame bytes: "
        <<image.size()
        <<"\n";


    std::ofstream file(
        "frame.raw",
        std::ios::binary
    );


    file.write(
        (char*)image.data(),
        image.size()
    );


    file.close();


    std::cout
        <<"Saved frame.raw\n";


    return !image.empty();

}