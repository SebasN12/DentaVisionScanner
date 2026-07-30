#include "GvspReceiver.h"

#include <iostream>


#ifdef _WIN32

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


    addr.sin_family =
        AF_INET;


    addr.sin_addr.s_addr =
        INADDR_ANY;


    addr.sin_port =
        htons(m_port);



    if(bind(
        m_socket,
        (sockaddr*)&addr,
        sizeof(addr))<0)
    {
        std::cout
            <<"Bind failed\n";

        return false;
    }



    std::cout
        <<"GVSP listening on port "
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


    int size =
        sizeof(sender);



    int received =
        recvfrom(
            m_socket,
            (char*)packet.data(),
            packet.size(),
            0,
            (sockaddr*)&sender,
            &size);



    if(received<=0)
    {
        return false;
    }



    packet.resize(received);



    return true;


#else

    return false;

#endif

}







uint32_t GvspReceiver::getBlockId(
    const std::vector<uint8_t>& packet)
{

    if(packet.size()<8)
        return 0;


    return
        (packet[2]<<8)
        |
        packet[3];

}






uint32_t GvspReceiver::getPacketId(
    const std::vector<uint8_t>& packet)
{

    if(packet.size()<8)
        return 0;


    return
        (packet[4]<<24)
        |
        (packet[5]<<16)
        |
        (packet[6]<<8)
        |
        packet[7];

}







bool GvspReceiver::receiveFrame(
    std::vector<uint8_t>& image)
{


    image.clear();


    uint32_t currentBlock=0;



    std::cout
        <<"Waiting frame...\n";



    while(true)
    {

        std::vector<uint8_t> packet;


        if(!receivePacket(packet))
            continue;



        uint32_t block =
            getBlockId(packet);



        uint32_t packetId =
            getPacketId(packet);



        std::cout
            <<"Block: "
            <<block
            <<" Packet: "
            <<packetId
            <<" Size: "
            <<packet.size()
            <<"\n";



        if(currentBlock==0)
        {
            currentBlock=block;
        }



        if(block!=currentBlock)
        {

            std::cout
            <<"New frame detected\n";


            break;
        }



        /*
            GVSP header:
            
            bytes 0-7
            
            payload starts at 8
        */


        for(size_t i=8;i<packet.size();i++)
        {
            image.push_back(packet[i]);
        }


    }



    std::cout
        <<"Frame size: "
        <<image.size()
        <<" bytes\n";


    return !image.empty();

}