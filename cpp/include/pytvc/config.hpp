#pragma once

#include <cstddef>

#ifndef PYTVC_SCALAR_DOUBLE
#define PYTVC_SCALAR_DOUBLE 1
#endif

namespace pytvc {

#if PYTVC_SCALAR_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

static_assert(sizeof(Scalar) == sizeof(float) || sizeof(Scalar) == sizeof(double),
              "pytvc::Scalar must be float or double");

constexpr Scalar kPi = Scalar(3.141592653589793238462643383279502884L);
constexpr Scalar kHalf = Scalar(0.5);
constexpr Scalar kTwo = Scalar(2.0);
constexpr Scalar kGravity = Scalar(-9.806);
constexpr Scalar kSeaLevelAirDensity = Scalar(1.225);

enum class Status {
    ok,
    full,
    empty,
    invalid_argument,
    not_found,
    non_finite,
};

constexpr bool ok(Status status) {
    return status == Status::ok;
}

}  // namespace pytvc
