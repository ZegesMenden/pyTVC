#pragma once

#include "pytvc/config.hpp"
#include "pytvc/math.hpp"

namespace pytvc {

class RigidBody {
public:
    RigidBody() = default;

    RigidBody(
        Scalar mass,
        const Vec3& inertia,
        const Vec3& position,
        const Vec3& velocity,
        const Quat& rotation,
        const Vec3& rot_vel)
        : mass_(mass),
          inv_mass_(mass > Scalar(0) ? Scalar(1) / mass : Scalar(0)),
          inertia_(inertia),
          position_(position),
          velocity_(velocity),
          rotation_(rotation.normalized()),
          rot_vel_(rot_vel) {}

    Scalar mass() const { return mass_; }
    Scalar inv_mass() const { return inv_mass_; }
    const Vec3& inertia() const { return inertia_; }
    const Vec3& position() const { return position_; }
    const Vec3& velocity() const { return velocity_; }
    const Quat& rotation() const { return rotation_; }
    const Vec3& rot_vel() const { return rot_vel_; }
    const Vec3& accel() const { return last_accel_; }

    Vec3& position() { return position_; }
    Vec3& velocity() { return velocity_; }
    Quat& rotation() { return rotation_; }
    Vec3& rot_vel() { return rot_vel_; }

    Status set_mass(Scalar mass) {
        if (mass <= Scalar(0)) {
            return Status::invalid_argument;
        }
        mass_ = mass;
        inv_mass_ = Scalar(1) / mass;
        return Status::ok;
    }

    Status set_inertia(const Vec3& inertia) {
        if (inertia.x == Scalar(0) || inertia.y == Scalar(0) || inertia.z == Scalar(0)) {
            return Status::invalid_argument;
        }
        inertia_ = inertia;
        return Status::ok;
    }

    void apply_torque(const Vec3& torque) {
        torque_accel_.x += torque.x / inertia_.x;
        torque_accel_.y += torque.y / inertia_.y;
        torque_accel_.z += torque.z / inertia_.z;
    }

    void apply_local_torque(const Vec3& torque) {
        const Vec3 world_torque = rotation_.rotate(torque);
        apply_torque(world_torque);
    }

    void apply_force(const Vec3& force, const Vec3& position) {
        accel_.x += force.x * inv_mass_;
        accel_.y += force.y * inv_mass_;
        accel_.z += force.z * inv_mass_;

        const Vec3 torque = position.cross(force);
        torque_accel_.x += torque.x / inertia_.x;
        torque_accel_.y += torque.y / inertia_.y;
        torque_accel_.z += torque.z / inertia_.z;
    }

    void apply_local_force(const Vec3& force, const Vec3& position) {
        const Vec3 world_force = rotation_.rotate(force);
        accel_.x += world_force.x * inv_mass_;
        accel_.y += world_force.y * inv_mass_;
        accel_.z += world_force.z * inv_mass_;

        const Vec3 local_torque = position.cross(force);
        const Vec3 world_torque = rotation_.rotate(local_torque);
        torque_accel_.x += world_torque.x / inertia_.x;
        torque_accel_.y += world_torque.y / inertia_.y;
        torque_accel_.z += world_torque.z / inertia_.z;
    }

    void update(Scalar dt) {
        velocity_.x += accel_.x * dt;
        velocity_.y += accel_.y * dt;
        velocity_.z += accel_.z * dt;

        position_.x += velocity_.x * dt;
        position_.y += velocity_.y * dt;
        position_.z += velocity_.z * dt;

        const Scalar wx = rot_vel_.x;
        const Scalar wy = rot_vel_.y;
        const Scalar wz = rot_vel_.z;
        const Scalar wmag2 = wx * wx + wy * wy + wz * wz;

        if (wmag2 > Scalar(0)) {
            const Scalar wmag = scalar_sqrt(wmag2);
            const Scalar inv = Scalar(1) / wmag;
            const Quat dq = Quat::from_axis_angle({wx * inv, wy * inv, wz * inv}, wmag * dt);
            rotation_ = (dq * rotation_).normalized();
        }

        rot_vel_.x += torque_accel_.x * dt;
        rot_vel_.y += torque_accel_.y * dt;
        rot_vel_.z += torque_accel_.z * dt;

        last_accel_ = accel_;
        torque_accel_.zero();
        accel_.zero();
    }

private:
    Scalar mass_ = Scalar(1);
    Scalar inv_mass_ = Scalar(1);
    Vec3 inertia_{Scalar(1), Scalar(1), Scalar(1)};
    Vec3 position_{};
    Vec3 velocity_{};
    Quat rotation_{};
    Vec3 rot_vel_{};
    Vec3 torque_accel_{};
    Vec3 accel_{};
    Vec3 last_accel_{};
};

}  // namespace pytvc
