#pragma once

class IScannerHardware
{
public:

    virtual bool connect() = 0;

    virtual bool disconnect() = 0;

    virtual bool projectPattern(int pattern) = 0;

    virtual bool captureImage() = 0;


    virtual ~IScannerHardware() = default;
};