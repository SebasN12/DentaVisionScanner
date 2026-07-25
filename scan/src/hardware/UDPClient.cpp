#include "UDPClient.h"

#include <iostream>


#ifdef _WIN32

#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.lib")

#endif

UDPClient::UDPClient(const std::string& remoteIp,
                     uint16_t remotePort,
                     uint16_t localPort)
    :
    m_remoteIp(remoteIp),
    m_remotePort(remotePort),
    m_localPort(localPort),
#ifdef _WIN32
    m_socket(INVALID_SOCKET)
#else
    m_socket(-1)
#endif
{
}


UDPClient::~UDPClient()
{
    close();
}


bool UDPClient::open()
{

#ifdef _WIN32


    WSADATA wsaData;


    if(WSAStartup(MAKEWORD(2,2), &wsaData) != 0)
    {
        std::cout 
            << "WSAStartup failed\n";

        return false;
    }

    m_socket = socket(
        AF_INET,
        SOCK_DGRAM,
        IPPROTO_UDP
    );


    if(m_socket == INVALID_SOCKET)
    {
        std::cout 
            << "Could not create socket\n";

        WSACleanup();

        return false;
    }

    DWORD timeout = 1000;

    setsockopt(
        m_socket,
        SOL_SOCKET,
        SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout),
        sizeof(timeout)
    );


    // Optional local port
    if(m_localPort != 0)
    {

        sockaddr_in local{};


        local.sin_family = AF_INET;

        local.sin_port = htons(m_localPort);

        local.sin_addr.s_addr = INADDR_ANY;


        if(bind(
            m_socket,
            reinterpret_cast<sockaddr*>(&local),
            sizeof(local)
        ) == SOCKET_ERROR)
        {

            std::cout 
                << "Bind failed\n";


            closesocket(m_socket);

            WSACleanup();

            return false;
        }

    }



    std::cout 
        << "UDP socket opened\n";


    return true;

#else

    return false;

#endif

}


void UDPClient::close()
{

#ifdef _WIN32


    if(m_socket != INVALID_SOCKET)
    {
        closesocket(m_socket);

        m_socket = INVALID_SOCKET;
    }


    WSACleanup();


#endif

}


bool UDPClient::sendPacket(
    const std::vector<uint8_t>& data)
{


#ifdef _WIN32


    sockaddr_in destination{};


    destination.sin_family = AF_INET;

    destination.sin_port = htons(m_remotePort);

    inet_pton(
        AF_INET,
        m_remoteIp.c_str(),
        &destination.sin_addr
    );



    int sent = sendto(
        m_socket,
        reinterpret_cast<const char*>(data.data()),
        static_cast<int>(data.size()),
        0,
        reinterpret_cast<sockaddr*>(&destination),
        sizeof(destination)
    );


    if(sent == SOCKET_ERROR)
    {

        std::cout
            << "sendto failed\n";

        return false;

    }

    std::cout
        << "Sent "
        << sent
        << " bytes\n";



    return true;

#else

    return false;

#endif

}



bool UDPClient::receivePacket(
    std::vector<uint8_t>& data,
    int timeoutMs)
{

#ifdef _WIN32

    DWORD timeout = timeoutMs;

    setsockopt(
        m_socket,
        SOL_SOCKET,
        SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout),
        sizeof(timeout)
    );

    data.resize(9000);

    sockaddr_in sender{};

    int senderLength = sizeof(sender);

    int received = recvfrom(
        m_socket,
        reinterpret_cast<char*>(data.data()),
        static_cast<int>(data.size()),
        0,
        reinterpret_cast<sockaddr*>(&sender),
        &senderLength
    );


    if(received == SOCKET_ERROR)
    {

        std::cout
            << "Receive timeout/error\n";

        return false;

    }

    data.resize(received);

    std::cout
        << "Received "
        << received
        << " bytes\n";

    return true;

#else

    return false;

#endif

}