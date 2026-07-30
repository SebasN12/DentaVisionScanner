#include "GvspReceiver.h"


#include <iostream>
#include <iomanip>


#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib,"Ws2_32.lib")

#endif



GvspReceiver::GvspReceiver(uint16_t port)
    :
    m_port(port),
    m_socket(INVALID_SOCKET)
{

}



GvspReceiver::~GvspReceiver()
{
    close();
}




bool GvspReceiver::open()
{

#ifdef _WIN32


    WSADATA wsa;


    if(WSAStartup(MAKEWORD(2,2),&wsa)!=0)
    {
        std::cout<<"WSAStartup failed\n";
        return false;
    }



    m_socket =
        socket(
            AF_INET,
            SOCK_DGRAM,
            IPPROTO_UDP
        );



    if(m_socket==INVALID_SOCKET)
    {
        std::cout<<"Socket creation failed\n";
        return false;
    }



    sockaddr_in local{};


    local.sin_family = AF_INET;

    local.sin_addr.s_addr =
        INADDR_ANY;

    local.sin_port =
        htons(m_port);



    if(bind(
        m_socket,
        reinterpret_cast<sockaddr*>(&local),
        sizeof(local)
    ) == SOCKET_ERROR)
    {

        std::cout
            <<"Bind failed. Is another program using this port?\n";


        closesocket(m_socket);

        return false;
    }



    std::cout
        <<"GVSP receiver listening on UDP port "
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


    if(m_socket!=INVALID_SOCKET)
    {
        closesocket(m_socket);
        m_socket=INVALID_SOCKET;
    }


    WSACleanup();


#endif

}







void GvspReceiver::start()
{


#ifdef _WIN32


    while(true)
    {


        std::vector<uint8_t> buffer(9000);



        sockaddr_in sender{};


        int senderSize =
            sizeof(sender);



        int received =
            recvfrom(
                m_socket,
                reinterpret_cast<char*>(buffer.data()),
                buffer.size(),
                0,
                reinterpret_cast<sockaddr*>(&sender),
                &senderSize
            );



        if(received==SOCKET_ERROR)
        {
            std::cout
                <<"Receive error\n";

            break;
        }



        buffer.resize(received);



        char ip[INET_ADDRSTRLEN];


        inet_ntop(
            AF_INET,
            &sender.sin_addr,
            ip,
            sizeof(ip)
        );



        uint16_t senderPort =
            ntohs(sender.sin_port);



        processPacket(
            buffer,
            ip,
            senderPort
        );


    }


#endif

}








void GvspReceiver::processPacket(
    const std::vector<uint8_t>& packet,
    const std::string& senderIp,
    uint16_t senderPort
)
{

    std::cout
        <<"\n-----------------------------\n";


    std::cout
        <<"Packet received\n";


    std::cout
        <<"From: "
        <<senderIp
        <<":"
        <<senderPort
        <<"\n";


    std::cout
        <<"Size: "
        <<packet.size()
        <<" bytes\n";



    analyzeGVSP(packet);


    printHex(packet);

}








void GvspReceiver::analyzeGVSP(
    const std::vector<uint8_t>& packet
)
{


    if(packet.size()<8)
    {
        std::cout
            <<"Packet too small\n";

        return;
    }



    /*
        GVSP header:

        Bytes 0-1:
            Status / Packet format

        Bytes 2-3:
            Block ID

        Bytes 4-7:
            Packet ID
    */


    uint16_t status =
        (packet[0]<<8)
        |
        packet[1];



    uint16_t block =
        (packet[2]<<8)
        |
        packet[3];



    uint32_t packetId =
        (packet[4]<<24)
        |
        (packet[5]<<16)
        |
        (packet[6]<<8)
        |
        packet[7];



    std::cout
        <<"GVSP Header\n";


    std::cout
        <<"Status/Type: 0x"
        <<std::hex
        <<status
        <<"\n";


    std::cout
        <<"Block ID: "
        <<std::dec
        <<block
        <<"\n";


    std::cout
        <<"Packet ID: "
        <<packetId
        <<"\n";

}








void GvspReceiver::printHex(
    const std::vector<uint8_t>& data
)
{

    std::cout
        <<"HEX dump:\n";


    size_t limit =
        std::min<size_t>(
            data.size(),
            64
        );



    for(size_t i=0;i<limit;i++)
    {

        std::cout
            <<std::hex
            <<std::setw(2)
            <<std::setfill('0')
            <<(int)data[i]
            <<" ";

    }


    std::cout
        <<std::dec
        <<"\n";

}