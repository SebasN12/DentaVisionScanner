#pragma once

#include <cstdint>
#include <vector>

class GvcpClient;

class GvcpSequence
{
public:

    enum class OperationType
    {
        Read,
        Write
    };

    struct Operation
    {
        OperationType type;
        uint32_t address;
        uint32_t value;
    };

    explicit GvcpSequence(GvcpClient& gvcp);

    void read(uint32_t address);

    void write(uint32_t address,
               uint32_t value);

    bool execute();

    void clear();

private:

    GvcpClient& m_gvcp;

    std::vector<Operation> m_operations;
};