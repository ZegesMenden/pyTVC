#pragma once

#include <cstddef>

#include "pytvc/actor.hpp"
#include "pytvc/fixed_vector.hpp"
#include "pytvc/rigid_body.hpp"
#include "pytvc/telemetry.hpp"

namespace pytvc {

struct ActorMount {
    Actor* actor = nullptr;
    Vec3 position{};
};

template <std::size_t MaxActors>
class Rocket {
public:
    Rocket() = default;

    Rocket(
        Scalar dry_mass,
        const Vec3& inertia,
        const Vec3& position,
        const Vec3& velocity,
        const Quat& rotation,
        const Vec3& rot_vel)
        : dry_mass_(dry_mass),
          body_(dry_mass, inertia, position, velocity, rotation, rot_vel) {}

    Scalar time() const { return time_; }
    Scalar dry_mass() const { return dry_mass_; }
    RigidBody& rigid_body() { return body_; }
    const RigidBody& rigid_body() const { return body_; }

    Status set_dry_mass(Scalar dry_mass) {
        if (dry_mass <= Scalar(0)) {
            return Status::invalid_argument;
        }
        dry_mass_ = dry_mass;
        return Status::ok;
    }

    Status add_actor(Actor* actor, const Vec3& position = {}) {
        if (actor == nullptr) {
            return Status::invalid_argument;
        }
        return actors_.push_back({actor, position});
    }

    std::size_t actor_count() const {
        return actors_.size();
    }

    Status step(Scalar dt, TelemetrySink* telemetry = nullptr) {
        if (dt <= Scalar(0)) {
            return Status::invalid_argument;
        }

        time_ += dt;

        Scalar actor_mass = Scalar(0);
        for (auto& mount : actors_) {
            mount.actor->update(body_, time_);
            actor_mass += mount.actor->mass();
        }

        body_.set_mass(dry_mass_ + actor_mass);

        for (auto& mount : actors_) {
            body_.apply_local_force(mount.actor->force(), mount.position);
            body_.apply_local_torque(mount.actor->torque());
        }

        body_.apply_force({kGravity * body_.mass(), Scalar(0), Scalar(0)}, {});

        if (body_.position().x <= Scalar(0)) {
            body_.position().x = Scalar(0);
            body_.velocity() = {};
        }

        body_.update(dt);

        if (telemetry != nullptr) {
            telemetry->record({
                time_,
                body_.position(),
                body_.velocity(),
                body_.accel(),
                body_.rotation(),
                body_.rot_vel(),
                body_.mass(),
            });
        }

        return Status::ok;
    }

private:
    Scalar dry_mass_ = Scalar(1);
    Scalar time_ = Scalar(0);
    RigidBody body_{};
    FixedVector<ActorMount, MaxActors> actors_{};
};

}  // namespace pytvc
