#include "pytvc/actor.hpp"
#include "pytvc/motor_curve.hpp"
#include "pytvc/rocket.hpp"

namespace {

class Aero final : public pytvc::AeroCoefficientModel {
public:
    pytvc::Scalar lift_coefficient(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return pytvc::Scalar(1);
    }

    pytvc::Scalar drag_coefficient(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return pytvc::Scalar(0.5);
    }
};

class Area final : public pytvc::AreaModel {
public:
    pytvc::Scalar area(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return pytvc::Scalar(1);
    }
};

}  // namespace

extern "C" int pytvc_embedded_compile_smoke() {
    pytvc::MotorCurve<8> motor;
    motor.add_point(pytvc::Scalar(0), pytvc::Scalar(0), pytvc::Scalar(100));
    motor.add_point(pytvc::Scalar(1), pytvc::Scalar(10), pytvc::Scalar(50));

    pytvc::MotorMount<1, 8> mount;
    mount.add_motor(&motor);
    mount.ignite(0, pytvc::Scalar(0));

    Aero aero;
    Area area;
    pytvc::ConstantPressure pressure;
    pytvc::RocketBody body_aero({}, &aero, &pressure, &area);

    pytvc::Rocket<4> rocket(
        pytvc::Scalar(2),
        {pytvc::Scalar(1), pytvc::Scalar(1), pytvc::Scalar(1)},
        {pytvc::Scalar(10), pytvc::Scalar(0), pytvc::Scalar(0)},
        {},
        {},
        {});

    rocket.add_actor(&mount, {});
    rocket.add_actor(&body_aero, {});
    rocket.step(pytvc::Scalar(0.01));

    return rocket.actor_count() == 2 ? 0 : 1;
}
