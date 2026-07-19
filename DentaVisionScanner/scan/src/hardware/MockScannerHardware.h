#pragma once

#include "IScannerHardware.h"


class MockScannerHardware : public IScannerHardware
{

public:

    bool connect() override;

    bool disconnect() override;

    bool projectPattern(int pattern) override;

    bool captureImage() override;

};