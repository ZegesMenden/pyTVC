#include <cmath>
#include <cstdio>

#include "pytvc/actor.hpp"
#include "pytvc/fixed_vector.hpp"
#include "pytvc/math.hpp"
#include "pytvc/motor_curve.hpp"
#include "pytvc/platform/output_sink.hpp"
#include "pytvc/rigid_body.hpp"
#include "pytvc/rocket.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

void near_scalar(pytvc::Scalar actual, pytvc::Scalar expected, pytvc::Scalar tol, const char* message) {
    if (std::fabs(actual - expected) > tol) {
        std::printf("FAIL: %s (actual=%g expected=%g)\n", message, static_cast<double>(actual), static_cast<double>(expected));
        ++failures;
    }
}

void near_vec(const pytvc::Vec3& actual, const pytvc::Vec3& expected, pytvc::Scalar tol, const char* message) {
    near_scalar(actual.x, expected.x, tol, message);
    near_scalar(actual.y, expected.y, tol, message);
    near_scalar(actual.z, expected.z, tol, message);
}

class ConstantAero final : public pytvc::AeroCoefficientModel {
public:
    ConstantAero(pytvc::Scalar cl, pytvc::Scalar cd) : cl_(cl), cd_(cd) {}
    pytvc::Scalar lift_coefficient(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return cl_;
    }
    pytvc::Scalar drag_coefficient(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return cd_;
    }

private:
    pytvc::Scalar cl_;
    pytvc::Scalar cd_;
};

class ConstantArea final : public pytvc::AreaModel {
public:
    explicit ConstantArea(pytvc::Scalar area) : area_(area) {}
    pytvc::Scalar area(pytvc::Scalar, pytvc::Scalar, const pytvc::RigidBody&) const override {
        return area_;
    }

private:
    pytvc::Scalar area_;
};

class SampleSink final : public pytvc::TelemetrySink {
public:
    void record(const pytvc::SimSample& sample) override {
        latest = sample;
        ++count;
    }

    pytvc::SimSample latest{};
    int count = 0;
};

void test_fixed_vector() {
    pytvc::FixedVector<int, 2> values;
    check(values.empty(), "fixed vector starts empty");
    check(pytvc::ok(values.push_back(1)), "fixed vector push first");
    check(pytvc::ok(values.push_back(2)), "fixed vector push second");
    check(values.full(), "fixed vector full");
    check(values.push_back(3) == pytvc::Status::full, "fixed vector rejects overflow");
    check(values.size() == 2, "fixed vector size retained after overflow");
    check(values[0] == 1 && values[1] == 2, "fixed vector preserves values");
}

void test_math() {
    using namespace pytvc;

    const Vec3 a{Scalar(1), Scalar(2), Scalar(3)};
    const Vec3 b{Scalar(4), Scalar(-5), Scalar(6)};
    near_vec(a + b, {Scalar(5), Scalar(-3), Scalar(9)}, Scalar(1e-9), "vec add");
    near_vec(a * b, {Scalar(4), Scalar(-10), Scalar(18)}, Scalar(1e-9), "vec multiply");
    near_scalar(a.dot(b), Scalar(12), Scalar(1e-9), "vec dot");
    near_vec(a.cross(b), {Scalar(27), Scalar(6), Scalar(-13)}, Scalar(1e-9), "vec cross");

    const Quat q = Quat::from_euler({Scalar(0), kPi / Scalar(2), Scalar(0)});
    near_vec(q.rotate({Scalar(1), Scalar(0), Scalar(0)}), {Scalar(0), Scalar(0), Scalar(-1)}, Scalar(1e-8), "quat rotate");
    near_vec(q.to_euler(), {Scalar(0), kPi / Scalar(2), Scalar(0)}, Scalar(1e-8), "quat euler round trip");
}

void test_motor_curve() {
    pytvc::MotorCurve<3> motor;
    check(pytvc::ok(motor.add_point(0.0, 0.0, 100.0)), "motor add point 0");
    check(pytvc::ok(motor.add_point(1.0, 10.0, 50.0)), "motor add point 1");
    check(pytvc::ok(motor.add_point(2.0, 0.0, 0.0)), "motor add point 2");
    check(motor.add_point(1.5, 0.0, 0.0) == pytvc::Status::full, "motor fixed capacity");
    near_scalar(motor.thrust_at(0.5), 5.0, 1e-9, "motor thrust interpolation");
    near_scalar(motor.mass_at(0.5), 75.0, 1e-9, "motor mass interpolation");
    near_scalar(motor.thrust_at(3.0), 0.0, 1e-9, "motor after burnout");
}

void test_rigid_body() {
    using namespace pytvc;

    RigidBody body(
        Scalar(2),
        {Scalar(1), Scalar(2), Scalar(4)},
        {},
        {},
        {},
        {});
    body.apply_force({Scalar(4), Scalar(0), Scalar(0)}, {Scalar(0), Scalar(1), Scalar(0)});
    body.update(Scalar(0.5));

    near_vec(body.velocity(), {Scalar(1), Scalar(0), Scalar(0)}, Scalar(1e-9), "rigid body velocity");
    near_vec(body.position(), {Scalar(0.5), Scalar(0), Scalar(0)}, Scalar(1e-9), "rigid body position");
    near_vec(body.rot_vel(), {Scalar(0), Scalar(0), Scalar(-0.5)}, Scalar(1e-9), "rigid body torque");

    RigidBody spinning(
        Scalar(1),
        {Scalar(1), Scalar(1), Scalar(1)},
        {},
        {},
        {},
        {Scalar(0), Scalar(0), kPi});
    spinning.update(Scalar(0.5));
    near_vec(spinning.rotation().rotate({Scalar(1), Scalar(0), Scalar(0)}), {Scalar(0), Scalar(1), Scalar(0)}, Scalar(1e-8), "rigid body rotation update");
}

void test_actors_and_rocket() {
    using namespace pytvc;

    ConstantAero aero(Scalar(2), Scalar(0.5));
    ConstantPressure pressure;
    ConstantArea area(Scalar(2));

    RigidBody body(
        Scalar(1),
        {Scalar(1), Scalar(1), Scalar(1)},
        {Scalar(10), Scalar(0), Scalar(0)},
        {Scalar(10), Scalar(0), Scalar(0)},
        {},
        {});

    Fin fin({Scalar(0), Scalar(1), Scalar(0)}, &aero, &pressure, Scalar(1), Scalar(0));
    fin.update(body, Scalar(0));
    near_vec(fin.force(), {Scalar(-30.625), Scalar(0), Scalar(122.5)}, Scalar(1e-8), "fin force");

    RigidBody body_side(
        Scalar(1),
        {Scalar(1), Scalar(1), Scalar(1)},
        {Scalar(10), Scalar(0), Scalar(0)},
        {Scalar(0), Scalar(10), Scalar(0)},
        {},
        {});
    RocketBody rocket_body({}, &aero, &pressure, &area);
    rocket_body.update(body_side, Scalar(0));
    near_vec(rocket_body.force(), {Scalar(0), Scalar(-306.25), Scalar(0)}, Scalar(1e-8), "rocket body aero force");

    MotorCurve<3> motor;
    motor.add_point(Scalar(0), Scalar(0), Scalar(100));
    motor.add_point(Scalar(1), Scalar(10), Scalar(50));
    motor.add_point(Scalar(2), Scalar(0), Scalar(0));

    MotorMount<1, 3> mount;
    check(ok(mount.add_motor(&motor)), "motor mount add motor");
    check(ok(mount.ignite(0, Scalar(0))), "motor mount ignite");
    mount.update(body, Scalar(0.5));
    near_vec(mount.force(), {Scalar(5), Scalar(0), Scalar(0)}, Scalar(1e-9), "motor mount force");
    near_scalar(mount.mass(), Scalar(0.075), Scalar(1e-9), "motor mount mass kg");

    Rocket<4> rocket(
        Scalar(2),
        {Scalar(1), Scalar(1), Scalar(1)},
        {Scalar(10), Scalar(0), Scalar(0)},
        {},
        {},
        {});
    RollDampener dampener({Scalar(-1), Scalar(-1), Scalar(-1)});
    check(ok(rocket.add_actor(&mount, {Scalar(0), Scalar(0), Scalar(0)})), "rocket add motor mount");
    check(ok(rocket.add_actor(&dampener, {})), "rocket add dampener");

    SampleSink sink;
    check(ok(rocket.step(Scalar(0.1), &sink)), "rocket step");
    check(sink.count == 1, "rocket telemetry record");
    check(rocket.actor_count() == 2, "rocket actor count");
    check(rocket.rigid_body().mass() > Scalar(2), "rocket actor mass contributes");
}

void test_output_sink_wrapper() {
    pytvc::BufferOutputSink<8> sink;
    const char* message = "abcdefghi";
    sink.write(message, 9);
    check(sink.size() == 7, "buffer output sink reserves terminator");
    check(sink.c_str()[0] == 'a' && sink.c_str()[6] == 'g' && sink.c_str()[7] == '\0', "buffer output sink contents");
}

}  // namespace

int main() {
    test_fixed_vector();
    test_math();
    test_motor_curve();
    test_rigid_body();
    test_actors_and_rocket();
    test_output_sink_wrapper();

    if (failures == 0) {
        std::printf("pytvc C++ core tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
