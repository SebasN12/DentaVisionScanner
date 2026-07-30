#include "GvcpSequence.h"

#include "GvcpClient.h"

#include <iomanip>
#include <iostream>

GvcpSequence::GvcpSequence(GvcpClient& gvcp)
    : m_gvcp(gvcp)
{
}

void GvcpSequence::read(uint32_t address)
{
    m_operations.push_back(
    {
        OperationType::Read,
        address,
        0
    });
}

void GvcpSequence::write(uint32_t address,
                         uint32_t value)
{
    m_operations.push_back(
    {
        OperationType::Write,
        address,
        value
    });
}

bool GvcpSequence::execute()
{
    std::cout << "\n=========================================\n";
    std::cout << "Executing GVCP sequence\n";
    std::cout << "Operations: " << m_operations.size() << "\n";
    std::cout << "=========================================\n";

    bool success = true;

    for(const auto& op : m_operations)
    {
        if(op.type == OperationType::Read)
        {
            uint32_t value = 0;

            bool ok =
                m_gvcp.readRegister(
                    op.address,
                    value);

            std::cout
                << "[READ ] 0x"
                << std::hex
                << std::uppercase
                << std::setw(8)
                << std::setfill('0')
                << op.address;

            if(ok)
            {
                std::cout
                    << " -> 0x"
                    << std::setw(8)
                    << value
                    << "   OK";
            }
            else
            {
                std::cout
                    << "   FAILED";

                success = false;
            }

            std::cout
                << std::dec
                << "\n";
        }
        else
        {
            bool ok =
                m_gvcp.writeRegister(
                    op.address,
                    op.value);

            std::cout
                << "[WRITE] 0x"
                << std::hex
                << std::uppercase
                << std::setw(8)
                << std::setfill('0')
                << op.address
                << " = 0x"
                << std::setw(8)
                << op.value;

            if(ok)
            {
                std::cout
                    << "   OK";
            }
            else
            {
                std::cout
                    << "   FAILED";

                success = false;
            }

            std::cout
                << std::dec
                << "\n";
        }
    }

    std::cout << "=========================================\n";

    return success;
}

void GvcpSequence::clear()
{
    m_operations.clear();
}