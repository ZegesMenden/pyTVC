#pragma once

#include "pytvc/config.hpp"
#include "pytvc/math.hpp"

namespace pytvc {

struct SimSample {
    Scalar time = Scalar(0);
    Vec3 position{};
    Vec3 velocity{};
    Vec3 accel{};
    Quat rotation{};
    Vec3 rot_vel{};
    Scalar mass = Scalar(0);
};

class TelemetrySink {
public:
    virtual ~TelemetrySink() = default;
    virtual void record(const SimSample& sample) = 0;
};

class NullTelemetrySink final : public TelemetrySink {
public:
    void record(const SimSample&) override {}
};

}  // namespace pytvc
