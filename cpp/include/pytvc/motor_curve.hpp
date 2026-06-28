#pragma once

#include <cstddef>

#include "pytvc/config.hpp"
#include "pytvc/fixed_vector.hpp"

namespace pytvc {

struct MotorPoint {
    Scalar time = Scalar(0);
    Scalar thrust = Scalar(0);
    Scalar mass = Scalar(0);
};

template <std::size_t MaxPoints>
class MotorCurve {
public:
    Status add_point(Scalar time, Scalar thrust, Scalar mass) {
        if (time < Scalar(0)) {
            return Status::invalid_argument;
        }
        if (points_.full()) {
            return Status::full;
        }
        if (!points_.empty() && time < points_[points_.size() - 1].time) {
            return Status::invalid_argument;
        }
        return points_.push_back({time, thrust, mass});
    }

    std::size_t size() const {
        return points_.size();
    }

    bool empty() const {
        return points_.empty();
    }

    const MotorPoint& point(std::size_t index) const {
        return points_[index];
    }

    Scalar burnout_time() const {
        if (points_.empty()) {
            return Scalar(0);
        }
        return points_[points_.size() - 1].time;
    }

    Scalar thrust_at(Scalar time) const {
        return interpolate(time, Field::thrust);
    }

    Scalar mass_at(Scalar time) const {
        return interpolate(time, Field::mass);
    }

private:
    enum class Field { thrust, mass };

    Scalar interpolate(Scalar time, Field field) const {
        if (points_.empty()) {
            return Scalar(0);
        }
        if (time < points_[0].time || time > points_[points_.size() - 1].time) {
            return Scalar(0);
        }

        for (std::size_t i = 0; i + 1 < points_.size(); ++i) {
            const MotorPoint& a = points_[i];
            const MotorPoint& b = points_[i + 1];
            if (a.time <= time && time <= b.time) {
                const Scalar denom = b.time - a.time;
                if (denom == Scalar(0)) {
                    return field == Field::thrust ? b.thrust : b.mass;
                }
                const Scalar u = (time - a.time) / denom;
                const Scalar av = field == Field::thrust ? a.thrust : a.mass;
                const Scalar bv = field == Field::thrust ? b.thrust : b.mass;
                return av + (bv - av) * u;
            }
        }
        return Scalar(0);
    }

    FixedVector<MotorPoint, MaxPoints> points_{};
};

}  // namespace pytvc
