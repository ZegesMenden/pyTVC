#pragma once

#include <cstddef>

#include "pytvc/config.hpp"
#include "pytvc/fixed_vector.hpp"
#include "pytvc/math.hpp"
#include "pytvc/motor_curve.hpp"
#include "pytvc/rigid_body.hpp"

namespace pytvc {

class Actor {
public:
    virtual ~Actor() = default;
    virtual void update(const RigidBody& body, Scalar time) {
        (void)body;
        (void)time;
    }
    virtual Vec3 force() const { return {}; }
    virtual Vec3 torque() const { return {}; }
    virtual Scalar mass() const { return Scalar(0); }
};

class RollDampener final : public Actor {
public:
    explicit RollDampener(const Vec3& damping_coefficient)
        : damping_coefficient_(damping_coefficient) {}

    void update(const RigidBody& body, Scalar) override {
        damping_torque_ = body.rotation().conjugate().rotate(damping_coefficient_ * body.rot_vel());
    }

    Vec3 torque() const override {
        return damping_torque_;
    }

private:
    Vec3 damping_coefficient_{};
    Vec3 damping_torque_{};
};

template <std::size_t MaxMotors, std::size_t MaxMotorPoints>
class MotorMount final : public Actor {
public:
    Status add_motor(const MotorCurve<MaxMotorPoints>* motor) {
        if (motor == nullptr) {
            return Status::invalid_argument;
        }
        const Status status = motors_.push_back(motor);
        if (!ok(status)) {
            return status;
        }
        return ignition_times_.push_back(Scalar(-1));
    }

    Status ignite(std::size_t motor_index, Scalar time) {
        if (motor_index >= motors_.size()) {
            return Status::not_found;
        }
        ignition_times_[motor_index] = time;
        return Status::ok;
    }

    void update(const RigidBody&, Scalar time) override {
        Scalar thrust = Scalar(0);
        Scalar mass_g = Scalar(0);

        for (std::size_t i = 0; i < motors_.size(); ++i) {
            const Scalar ignition_time = ignition_times_[i];
            const Scalar motor_time = ignition_time == Scalar(-1) ? Scalar(0) : time - ignition_time;
            thrust += motors_[i]->thrust_at(motor_time);
            mass_g += motors_[i]->mass_at(motor_time);
        }

        thrust_vec_ = {thrust, Scalar(0), Scalar(0)};
        mass_kg_ = mass_g / Scalar(1000);
    }

    Vec3 force() const override {
        return thrust_vec_;
    }

    Scalar mass() const override {
        return mass_kg_;
    }

private:
    FixedVector<const MotorCurve<MaxMotorPoints>*, MaxMotors> motors_{};
    FixedVector<Scalar, MaxMotors> ignition_times_{};
    Vec3 thrust_vec_{};
    Scalar mass_kg_ = Scalar(0);
};

class ServoModel {
public:
    virtual ~ServoModel() = default;
    virtual Vec3 transfer(
        const RigidBody& body,
        Scalar time,
        const Vec3& target_angles,
        Scalar thrust,
        Scalar motor_mass_g) const = 0;
};

class LinkageModel {
public:
    virtual ~LinkageModel() = default;
    virtual Quat linkage(const RigidBody& body, Scalar time, const Vec3& servo_output) const = 0;
};

template <std::size_t MaxMotors, std::size_t MaxMotorPoints>
class TVCMount final : public Actor {
public:
    TVCMount() = default;

    TVCMount(const ServoModel* servo, const LinkageModel* linkage)
        : servo_(servo), linkage_(linkage) {}

    void set_servo_model(const ServoModel* servo) {
        servo_ = servo;
    }

    void set_linkage_model(const LinkageModel* linkage) {
        linkage_ = linkage;
    }

    Status add_motor(const MotorCurve<MaxMotorPoints>* motor) {
        if (motor == nullptr) {
            return Status::invalid_argument;
        }
        const Status status = motors_.push_back(motor);
        if (!ok(status)) {
            return status;
        }
        return ignition_times_.push_back(Scalar(-1));
    }

    Status ignite(std::size_t motor_index, Scalar time) {
        if (motor_index >= motors_.size()) {
            return Status::not_found;
        }
        ignition_times_[motor_index] = time;
        return Status::ok;
    }

    void set_target_angles(const Vec3& angles) {
        target_angles_ = angles;
    }

    const Vec3& target_angles() const {
        return target_angles_;
    }

    const Quat& angles() const {
        return angles_;
    }

    void update(const RigidBody& body, Scalar time) override {
        Scalar thrust = Scalar(0);
        Scalar mass_g = Scalar(0);

        for (std::size_t i = 0; i < motors_.size(); ++i) {
            const Scalar ignition_time = ignition_times_[i];
            const Scalar motor_time = ignition_time == Scalar(-1) ? Scalar(0) : time - ignition_time;
            thrust += motors_[i]->thrust_at(motor_time);
            mass_g += motors_[i]->mass_at(motor_time);
        }

        mass_kg_ = mass_g / Scalar(1000);

        Vec3 servo_output = target_angles_;
        if (servo_ != nullptr) {
            servo_output = servo_->transfer(body, time, target_angles_, thrust, mass_g);
        }

        angles_ = linkage_ != nullptr
            ? linkage_->linkage(body, time, servo_output)
            : Quat::from_euler(servo_output);
        thrust_vec_ = angles_.rotate({thrust, Scalar(0), Scalar(0)});
    }

    Vec3 force() const override {
        return thrust_vec_;
    }

    Scalar mass() const override {
        return mass_kg_;
    }

private:
    FixedVector<const MotorCurve<MaxMotorPoints>*, MaxMotors> motors_{};
    FixedVector<Scalar, MaxMotors> ignition_times_{};
    const ServoModel* servo_ = nullptr;
    const LinkageModel* linkage_ = nullptr;
    Vec3 target_angles_{};
    Quat angles_{};
    Vec3 thrust_vec_{};
    Scalar mass_kg_ = Scalar(0);
};

class AeroCoefficientModel {
public:
    virtual ~AeroCoefficientModel() = default;
    virtual Scalar lift_coefficient(Scalar aoa, Scalar flow_speed, const RigidBody& state) const = 0;
    virtual Scalar drag_coefficient(Scalar aoa, Scalar flow_speed, const RigidBody& state) const = 0;
};

class PressureModel {
public:
    virtual ~PressureModel() = default;
    virtual Scalar pressure(const RigidBody& state) const = 0;
};

class AreaModel {
public:
    virtual ~AreaModel() = default;
    virtual Scalar area(Scalar aoa, Scalar flow_speed, const RigidBody& state) const = 0;
};

class ConstantPressure final : public PressureModel {
public:
    explicit ConstantPressure(Scalar pressure = kSeaLevelAirDensity) : pressure_(pressure) {}
    Scalar pressure(const RigidBody&) const override { return pressure_; }

private:
    Scalar pressure_ = kSeaLevelAirDensity;
};

class Fin : public Actor {
public:
    Fin() = default;

    Fin(
        const Vec3& position,
        const AeroCoefficientModel* coefficients,
        const PressureModel* pressure,
        Scalar area,
        Scalar body_angle)
        : position_(position),
          coefficients_(coefficients),
          pressure_(pressure),
          area_(area),
          body_angle_(body_angle) {}

    const Vec3& position() const { return position_; }
    Scalar angle() const { return angle_; }
    Scalar body_angle() const { return body_angle_; }
    const Vec3& last_lift_body() const { return lift_body_; }
    const Vec3& last_drag_body() const { return drag_body_; }

    void set_angle(Scalar angle) {
        angle_ = angle;
    }

    Quat rotation_body() const {
        return Quat::from_euler({body_angle_, Scalar(0), Scalar(0)})
            * Quat::from_euler({Scalar(0), angle_, Scalar(0)});
    }

    void update(const RigidBody& body, Scalar) override {
        force_ = {};
        torque_ = {};
        lift_body_ = {};
        drag_body_ = {};

        if (coefficients_ == nullptr || pressure_ == nullptr || area_ <= Scalar(0)) {
            return;
        }

        Vec3 body_velocity = body.rotation().conjugate().rotate(body.velocity());
        const Vec3 body_rot_vel = body.rotation().conjugate().rotate(body.rot_vel());
        body_velocity += body_rot_vel.cross(position_);

        const Quat fin_rotation = rotation_body();
        const Vec3 fin_velocity = fin_rotation.conjugate().rotate(body_velocity);
        const Scalar flow_speed = fin_velocity.len();
        if (flow_speed == Scalar(0)) {
            return;
        }

        const Scalar aoa = scalar_atan2(-fin_velocity.z, fin_velocity.x);
        const Scalar cl = coefficients_->lift_coefficient(aoa, flow_speed, body);
        const Scalar cd = coefficients_->drag_coefficient(aoa, flow_speed, body);

        const Vec3 flow_dir = fin_velocity.normalized();
        const Vec3 drag_dir = flow_dir * Scalar(-1);
        const Vec3 span_axis{Scalar(0), Scalar(1), Scalar(0)};
        const Vec3 lift_dir = flow_dir.cross(span_axis).normalized();
        const Scalar aero = Scalar(0.5) * pressure_->pressure(body) * flow_speed * flow_speed * area_;

        const Vec3 lift = lift_dir * aero * cl;
        const Vec3 drag = drag_dir * aero * cd;

        lift_body_ = fin_rotation.rotate(lift);
        drag_body_ = fin_rotation.rotate(drag);
        force_ = fin_rotation.rotate(lift + drag);
    }

    Vec3 force() const override {
        return force_;
    }

    Vec3 torque() const override {
        return torque_;
    }

private:
    Vec3 position_{};
    const AeroCoefficientModel* coefficients_ = nullptr;
    const PressureModel* pressure_ = nullptr;
    Scalar area_ = Scalar(0);
    Scalar angle_ = Scalar(0);
    Scalar body_angle_ = Scalar(0);
    Vec3 lift_body_{};
    Vec3 drag_body_{};
    Vec3 force_{};
    Vec3 torque_{};
};

template <std::size_t MaxFins>
class FinCan : public Actor {
public:
    const Vec3& position() const { return position_; }
    std::size_t fin_count() const { return fins_.size(); }

    Status configure(
        const Vec3& position,
        const AeroCoefficientModel* coefficients,
        const PressureModel* pressure,
        Scalar area,
        Scalar radial_distance,
        std::size_t fin_count,
        Scalar initial_offset = Scalar(0)) {
        if (fin_count > MaxFins || fin_count == 0) {
            return Status::invalid_argument;
        }
        position_ = position;
        fins_.clear();
        for (std::size_t i = 0; i < fin_count; ++i) {
            const Scalar angle = (Scalar(2) * kPi) * (Scalar(i) / Scalar(fin_count));
            Fin fin(
                Quat::from_euler({angle, Scalar(0), Scalar(0)}).rotate({Scalar(0), radial_distance, Scalar(0)}) + position,
                coefficients,
                pressure,
                area,
                angle);
            fin.set_angle(initial_offset);
            const Status status = fins_.push_back(fin);
            if (!ok(status)) {
                return status;
            }
        }
        return Status::ok;
    }

    Fin& fin(std::size_t index) { return fins_[index]; }
    const Fin& fin(std::size_t index) const { return fins_[index]; }

    void update(const RigidBody& body, Scalar time) override {
        for (auto& fin : fins_) {
            fin.update(body, time);
        }
    }

    Vec3 force() const override {
        Vec3 total{};
        for (const auto& fin : fins_) {
            total += fin.force();
        }
        return total;
    }

    Vec3 torque() const override {
        Vec3 total{};
        for (const auto& fin : fins_) {
            total += fin.torque();
            total += fin.position().cross(fin.force()) * Vec3{Scalar(1), Scalar(0), Scalar(0)};
        }
        return total;
    }

protected:
    FixedVector<Fin, MaxFins> fins_{};
    Vec3 position_{};
};

template <std::size_t MaxFins>
class SpinCan final : public FinCan<MaxFins> {
public:
    void set_roll_coefficient(Scalar roll_coefficient) {
        roll_coefficient_ = roll_coefficient;
    }

    Vec3 torque() const override {
        Vec3 total{};
        for (const auto& fin : this->fins_) {
            total += fin.torque();
            total += fin.position().cross(fin.force()) * Vec3{roll_coefficient_, Scalar(0), Scalar(0)};
        }
        return total;
    }

private:
    Scalar roll_coefficient_ = Scalar(0);
};

class Airbrake : public Actor {
public:
    Airbrake() = default;

    Airbrake(
        const Vec3& position,
        Scalar body_angle,
        Scalar area,
        const AeroCoefficientModel* coefficients,
        Scalar max_angle = Scalar(60) * kPi / Scalar(180))
        : position_(position),
          body_angle_(body_angle),
          area_(area),
          coefficients_(coefficients),
          max_angle_(max_angle) {}

    const Vec3& position() const { return position_; }
    Scalar angle() const { return angle_; }
    Scalar body_angle() const { return body_angle_; }
    const Vec3& last_drag() const { return last_drag_; }

    void set_angle(Scalar angle) {
        angle_ = clamp(angle, Scalar(0), max_angle_);
    }

    void update(const RigidBody& body, Scalar) override {
        force_ = {};
        torque_ = {};
        last_drag_ = {};

        if (coefficients_ == nullptr || area_ <= Scalar(0)) {
            return;
        }

        Vec3 body_velocity = body.rotation().conjugate().rotate(body.velocity());
        const Vec3 body_rot_vel = body.rotation().conjugate().rotate(body.rot_vel());
        body_velocity += body_rot_vel.cross(position_);

        const Quat brake_rotation =
            Quat::from_euler({body_angle_, Scalar(0), Scalar(0)})
            * Quat::from_euler({Scalar(0), angle_, Scalar(0)});
        const Vec3 brake_velocity = brake_rotation.conjugate().rotate(body_velocity);
        const Scalar flow_speed = brake_velocity.len();
        if (flow_speed <= Scalar(0)) {
            return;
        }

        const Scalar effective_area = area_ * scalar_sin(angle_ < Scalar(0) ? -angle_ : angle_);
        if (effective_area <= Scalar(0)) {
            return;
        }

        const Vec3 drag_direction = brake_velocity.normalized() * Scalar(-1);
        const Scalar cd = coefficients_->drag_coefficient(angle_, flow_speed, body);
        const Scalar drag_mag = Scalar(0.5) * kSeaLevelAirDensity * flow_speed * flow_speed * effective_area * cd;
        const Vec3 drag_fin_frame = drag_direction * drag_mag;
        last_drag_ = brake_rotation.rotate(drag_fin_frame);
        force_ = last_drag_;
    }

    Vec3 force() const override {
        return force_;
    }

    Vec3 torque() const override {
        return torque_;
    }

private:
    Vec3 position_{};
    Scalar body_angle_ = Scalar(0);
    Scalar area_ = Scalar(0);
    const AeroCoefficientModel* coefficients_ = nullptr;
    Scalar max_angle_ = Scalar(60) * kPi / Scalar(180);
    Scalar angle_ = Scalar(0);
    Vec3 force_{};
    Vec3 torque_{};
    Vec3 last_drag_{};
};

template <std::size_t MaxBrakes>
class AirbrakeCan final : public Actor {
public:
    const Vec3& position() const { return position_; }
    std::size_t brake_count() const { return brakes_.size(); }

    Status configure(
        const Vec3& position,
        const AeroCoefficientModel* coefficients,
        Scalar area,
        Scalar radial_distance,
        std::size_t brake_count,
        Scalar initial_offset = Scalar(0),
        Scalar max_angle = Scalar(60) * kPi / Scalar(180)) {
        if (brake_count > MaxBrakes || brake_count == 0) {
            return Status::invalid_argument;
        }
        position_ = position;
        brakes_.clear();
        for (std::size_t i = 0; i < brake_count; ++i) {
            const Scalar angle = (Scalar(2) * kPi) * (Scalar(i) / Scalar(brake_count));
            Airbrake brake(
                Quat::from_euler({angle, Scalar(0), Scalar(0)}).rotate({Scalar(0), radial_distance, Scalar(0)}) + position,
                angle,
                area,
                coefficients,
                max_angle);
            brake.set_angle(initial_offset);
            const Status status = brakes_.push_back(brake);
            if (!ok(status)) {
                return status;
            }
        }
        return Status::ok;
    }

    Airbrake& brake(std::size_t index) { return brakes_[index]; }
    const Airbrake& brake(std::size_t index) const { return brakes_[index]; }

    void set_angle(Scalar angle) {
        for (auto& brake : brakes_) {
            brake.set_angle(angle);
        }
    }

    void update(const RigidBody& body, Scalar time) override {
        for (auto& brake : brakes_) {
            brake.update(body, time);
        }
    }

    Vec3 force() const override {
        Vec3 total{};
        for (const auto& brake : brakes_) {
            total += brake.force();
        }
        return total;
    }

    Vec3 torque() const override {
        Vec3 total{};
        for (const auto& brake : brakes_) {
            total += brake.position().cross(brake.force());
        }
        return total;
    }

private:
    FixedVector<Airbrake, MaxBrakes> brakes_{};
    Vec3 position_{};
};

class RocketBody final : public Actor {
public:
    RocketBody(
        const Vec3& position,
        const AeroCoefficientModel* coefficients,
        const PressureModel* pressure,
        const AreaModel* area)
        : position_(position), coefficients_(coefficients), pressure_(pressure), area_(area) {}

    void update(const RigidBody& body, Scalar) override {
        lift_ = {};
        drag_ = {};

        if (coefficients_ == nullptr || pressure_ == nullptr || area_ == nullptr) {
            return;
        }

        const Vec3 air_velocity = body.velocity();
        const Scalar speed = air_velocity.len();
        if (speed <= Scalar(0)) {
            return;
        }

        const Vec3 body_axis_world = body.rotation().rotate({Scalar(1), Scalar(0), Scalar(0)}).normalized();
        Scalar aoa = air_velocity.angle_between(body_axis_world);
        if (aoa > kPi / Scalar(2)) {
            aoa = kPi - aoa;
        }

        const Scalar cd = coefficients_->drag_coefficient(aoa, speed, body);
        const Scalar cl = coefficients_->lift_coefficient(aoa, speed, body);
        const Scalar ref_area = area_->area(aoa, speed, body);
        const Scalar pressure = pressure_->pressure(body);
        const Scalar lift_mag = Scalar(0.5) * cl * ref_area * pressure * speed * speed;
        const Scalar drag_mag = Scalar(0.5) * cd * ref_area * pressure * speed * speed;

        drag_ = air_velocity.normalized() * -drag_mag;

        const Vec3 v_para = body_axis_world * air_velocity.dot(body_axis_world);
        const Vec3 v_perp = air_velocity - v_para;
        if (v_perp.len() > Scalar(1e-12) && lift_mag != Scalar(0)) {
            lift_ = v_perp.normalized() * -lift_mag;
        }
    }

    Vec3 force() const override {
        return lift_ + drag_;
    }

    const Vec3& lift() const { return lift_; }
    const Vec3& drag() const { return drag_; }

private:
    Vec3 position_{};
    const AeroCoefficientModel* coefficients_ = nullptr;
    const PressureModel* pressure_ = nullptr;
    const AreaModel* area_ = nullptr;
    Vec3 lift_{};
    Vec3 drag_{};
};

}  // namespace pytvc
