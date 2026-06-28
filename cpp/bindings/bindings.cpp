#include <memory>
#include <sstream>
#include <stdexcept>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "pytvc/actor.hpp"
#include "pytvc/math.hpp"
#include "pytvc/motor_curve.hpp"
#include "pytvc/rigid_body.hpp"
#include "pytvc/rocket.hpp"
#include "pytvc/telemetry.hpp"

namespace py = pybind11;

namespace {

using PyMotorCurve = pytvc::MotorCurve<1024>;
using PyMotorMount = pytvc::MotorMount<8, 1024>;
using PyTVCMount = pytvc::TVCMount<8, 1024>;
using PyRocket = pytvc::Rocket<64>;
using PyFinCan = pytvc::FinCan<16>;
using PySpinCan = pytvc::SpinCan<16>;
using PyAirbrakeCan = pytvc::AirbrakeCan<16>;

void throw_on_bad(pytvc::Status status) {
    switch (status) {
        case pytvc::Status::ok:
            return;
        case pytvc::Status::full:
            throw py::value_error("fixed-capacity container is full");
        case pytvc::Status::empty:
            throw py::value_error("fixed-capacity container is empty");
        case pytvc::Status::invalid_argument:
            throw py::value_error("invalid argument");
        case pytvc::Status::not_found:
            throw py::index_error("item not found");
        case pytvc::Status::non_finite:
            throw py::value_error("non-finite value");
    }
    throw std::runtime_error("unknown pytvc status");
}

bool vec_equal(const pytvc::Vec3& a, const pytvc::Vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool quat_equal(const pytvc::Quat& a, const pytvc::Quat& b) {
    return a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z;
}

class PyActor : public pytvc::Actor {
public:
    using pytvc::Actor::Actor;

    void update(const pytvc::RigidBody& body, pytvc::Scalar time) override {
        PYBIND11_OVERRIDE(void, pytvc::Actor, update, body, time);
    }

    pytvc::Vec3 force() const override {
        PYBIND11_OVERRIDE(pytvc::Vec3, pytvc::Actor, force);
    }

    pytvc::Vec3 torque() const override {
        PYBIND11_OVERRIDE(pytvc::Vec3, pytvc::Actor, torque);
    }

    pytvc::Scalar mass() const override {
        PYBIND11_OVERRIDE(pytvc::Scalar, pytvc::Actor, mass);
    }
};

class PyAeroCoefficientModel : public pytvc::AeroCoefficientModel {
public:
    using pytvc::AeroCoefficientModel::AeroCoefficientModel;

    pytvc::Scalar lift_coefficient(
        pytvc::Scalar aoa,
        pytvc::Scalar flow_speed,
        const pytvc::RigidBody& state) const override {
        PYBIND11_OVERRIDE_PURE(
            pytvc::Scalar,
            pytvc::AeroCoefficientModel,
            lift_coefficient,
            aoa,
            flow_speed,
            state);
    }

    pytvc::Scalar drag_coefficient(
        pytvc::Scalar aoa,
        pytvc::Scalar flow_speed,
        const pytvc::RigidBody& state) const override {
        PYBIND11_OVERRIDE_PURE(
            pytvc::Scalar,
            pytvc::AeroCoefficientModel,
            drag_coefficient,
            aoa,
            flow_speed,
            state);
    }
};

class PyServoModel : public pytvc::ServoModel {
public:
    using pytvc::ServoModel::ServoModel;

    pytvc::Vec3 transfer(
        const pytvc::RigidBody& body,
        pytvc::Scalar time,
        const pytvc::Vec3& target_angles,
        pytvc::Scalar thrust,
        pytvc::Scalar motor_mass_g) const override {
        PYBIND11_OVERRIDE_PURE(
            pytvc::Vec3,
            pytvc::ServoModel,
            transfer,
            body,
            time,
            target_angles,
            thrust,
            motor_mass_g);
    }
};

class PyLinkageModel : public pytvc::LinkageModel {
public:
    using pytvc::LinkageModel::LinkageModel;

    pytvc::Quat linkage(
        const pytvc::RigidBody& body,
        pytvc::Scalar time,
        const pytvc::Vec3& servo_output) const override {
        PYBIND11_OVERRIDE_PURE(
            pytvc::Quat,
            pytvc::LinkageModel,
            linkage,
            body,
            time,
            servo_output);
    }
};

class PyPressureModel : public pytvc::PressureModel {
public:
    using pytvc::PressureModel::PressureModel;

    pytvc::Scalar pressure(const pytvc::RigidBody& state) const override {
        PYBIND11_OVERRIDE_PURE(pytvc::Scalar, pytvc::PressureModel, pressure, state);
    }
};

class PyAreaModel : public pytvc::AreaModel {
public:
    using pytvc::AreaModel::AreaModel;

    pytvc::Scalar area(
        pytvc::Scalar aoa,
        pytvc::Scalar flow_speed,
        const pytvc::RigidBody& state) const override {
        PYBIND11_OVERRIDE_PURE(pytvc::Scalar, pytvc::AreaModel, area, aoa, flow_speed, state);
    }
};

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

class PythonTelemetrySink final : public pytvc::TelemetrySink {
public:
    explicit PythonTelemetrySink(py::object callback) : callback_(callback) {}

    void record(const pytvc::SimSample& sample) override {
        callback_(sample);
    }

private:
    py::object callback_;
};

}  // namespace

PYBIND11_MODULE(_pytvc_cpp, m) {
    m.doc() = "Host-only Python bindings for the embedded pyTVC C++ core";
    m.attr("SCALAR_DOUBLE") = py::bool_(PYTVC_SCALAR_DOUBLE != 0);

    py::enum_<pytvc::Status>(m, "Status")
        .value("ok", pytvc::Status::ok)
        .value("full", pytvc::Status::full)
        .value("empty", pytvc::Status::empty)
        .value("invalid_argument", pytvc::Status::invalid_argument)
        .value("not_found", pytvc::Status::not_found)
        .value("non_finite", pytvc::Status::non_finite);

    py::class_<pytvc::Vec3>(m, "Vector3")
        .def(py::init<pytvc::Scalar, pytvc::Scalar, pytvc::Scalar>(),
             py::arg("x") = 0.0,
             py::arg("y") = 0.0,
             py::arg("z") = 0.0)
        .def_readwrite("x", &pytvc::Vec3::x)
        .def_readwrite("y", &pytvc::Vec3::y)
        .def_readwrite("z", &pytvc::Vec3::z)
        .def("copy", [](const pytvc::Vec3& self) { return self; })
        .def("set", [](pytvc::Vec3& self, pytvc::Scalar x, pytvc::Scalar y, pytvc::Scalar z) -> pytvc::Vec3& {
            self.x = x;
            self.y = y;
            self.z = z;
            return self;
        }, py::return_value_policy::reference_internal)
        .def("zero", &pytvc::Vec3::zero, py::return_value_policy::reference_internal)
        .def("copy_from", [](pytvc::Vec3& self, const pytvc::Vec3& other) -> pytvc::Vec3& {
            self = other;
            return self;
        }, py::return_value_policy::reference_internal)
        .def("add_scaled", &pytvc::Vec3::add_scaled, py::return_value_policy::reference_internal)
        .def("mag2", &pytvc::Vec3::mag2)
        .def("len", &pytvc::Vec3::len)
        .def("norm", &pytvc::Vec3::normalized)
        .def("dot", &pytvc::Vec3::dot)
        .def("cross", &pytvc::Vec3::cross)
        .def("angleBetween", &pytvc::Vec3::angle_between)
        .def("angle_between", &pytvc::Vec3::angle_between)
        .def("__abs__", &pytvc::Vec3::len)
        .def("__iter__", [](const pytvc::Vec3& self) {
            return py::iter(py::make_tuple(self.x, self.y, self.z));
        })
        .def("__getitem__", [](const pytvc::Vec3& self, int index) {
            if (index == 0) return self.x;
            if (index == 1) return self.y;
            if (index == 2) return self.z;
            throw py::index_error("Vector3 index out of range");
        })
        .def("__repr__", [](const pytvc::Vec3& self) {
            std::ostringstream out;
            out << "Vector3(" << self.x << ", " << self.y << ", " << self.z << ")";
            return out.str();
        })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self * pytvc::Scalar())
        .def(pytvc::Scalar() * py::self)
        .def(py::self / py::self)
        .def(py::self / pytvc::Scalar())
        .def("__eq__", &vec_equal)
        .def("__ne__", [](const pytvc::Vec3& a, const pytvc::Vec3& b) { return !vec_equal(a, b); });

    m.attr("Vec3") = m.attr("Vector3");

    py::class_<pytvc::Quat>(m, "Quaternion")
        .def(py::init<pytvc::Scalar, pytvc::Scalar, pytvc::Scalar, pytvc::Scalar>(),
             py::arg("w") = 1.0,
             py::arg("x") = 0.0,
             py::arg("y") = 0.0,
             py::arg("z") = 0.0)
        .def_readwrite("w", &pytvc::Quat::w)
        .def_readwrite("x", &pytvc::Quat::x)
        .def_readwrite("y", &pytvc::Quat::y)
        .def_readwrite("z", &pytvc::Quat::z)
        .def("copy", [](const pytvc::Quat& self) { return self; })
        .def("set", [](pytvc::Quat& self, pytvc::Scalar w, pytvc::Scalar x, pytvc::Scalar y, pytvc::Scalar z) -> pytvc::Quat& {
            self.w = w;
            self.x = x;
            self.y = y;
            self.z = z;
            return self;
        }, py::return_value_policy::reference_internal)
        .def("conjugate", &pytvc::Quat::conjugate)
        .def("mag2", &pytvc::Quat::mag2)
        .def("len", &pytvc::Quat::len)
        .def("norm", &pytvc::Quat::normalized)
        .def("normalize_ip", &pytvc::Quat::normalize_in_place, py::return_value_policy::reference_internal)
        .def("rotate", &pytvc::Quat::rotate)
        .def("rotateSafe", &pytvc::Quat::rotate_safe)
        .def("rotate_safe", &pytvc::Quat::rotate_safe)
        .def("dot", &pytvc::Quat::dot)
        .def_property_readonly("xyz", &pytvc::Quat::xyz)
        .def_static("fromEulerAngles", &pytvc::Quat::from_euler)
        .def_static("from_euler", &pytvc::Quat::from_euler)
        .def_static("fromAxisAngle", &pytvc::Quat::from_axis_angle)
        .def_static("from_axis_angle", &pytvc::Quat::from_axis_angle)
        .def("toEulerAngles", &pytvc::Quat::to_euler)
        .def("to_euler", &pytvc::Quat::to_euler)
        .def("__abs__", &pytvc::Quat::len)
        .def("__iter__", [](const pytvc::Quat& self) {
            return py::iter(py::make_tuple(self.w, self.x, self.y, self.z));
        })
        .def("__getitem__", [](const pytvc::Quat& self, int index) {
            if (index == 0) return self.w;
            if (index == 1) return self.x;
            if (index == 2) return self.y;
            if (index == 3) return self.z;
            throw py::index_error("Quaternion index out of range");
        })
        .def("__repr__", [](const pytvc::Quat& self) {
            std::ostringstream out;
            out << "Quaternion(" << self.w << ", " << self.x << ", " << self.y << ", " << self.z << ")";
            return out.str();
        })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self * pytvc::Scalar())
        .def(pytvc::Scalar() * py::self)
        .def(py::self / pytvc::Scalar())
        .def("__eq__", &quat_equal)
        .def("__ne__", [](const pytvc::Quat& a, const pytvc::Quat& b) { return !quat_equal(a, b); });

    m.attr("Quat") = m.attr("Quaternion");

    py::class_<pytvc::RigidBody>(m, "RigidBody")
        .def(py::init<
                 pytvc::Scalar,
                 const pytvc::Vec3&,
                 const pytvc::Vec3&,
                 const pytvc::Vec3&,
                 const pytvc::Quat&,
                 const pytvc::Vec3&>(),
             py::arg("mass"),
             py::arg("inertia"),
             py::arg("position"),
             py::arg("velocity"),
             py::arg("rotation"),
             py::arg("rotVel"))
        .def_property("mass", &pytvc::RigidBody::mass, [](pytvc::RigidBody& self, pytvc::Scalar mass) {
            throw_on_bad(self.set_mass(mass));
        })
        .def_property_readonly("inv_mass", &pytvc::RigidBody::inv_mass)
        .def_property("inertia", [](pytvc::RigidBody& self) { return self.inertia(); }, [](pytvc::RigidBody& self, const pytvc::Vec3& inertia) {
            throw_on_bad(self.set_inertia(inertia));
        })
        .def_property("position", [](pytvc::RigidBody& self) -> pytvc::Vec3& { return self.position(); }, [](pytvc::RigidBody& self, const pytvc::Vec3& v) {
            self.position() = v;
        }, py::return_value_policy::reference_internal)
        .def_property("velocity", [](pytvc::RigidBody& self) -> pytvc::Vec3& { return self.velocity(); }, [](pytvc::RigidBody& self, const pytvc::Vec3& v) {
            self.velocity() = v;
        }, py::return_value_policy::reference_internal)
        .def_property("rotation", [](pytvc::RigidBody& self) -> pytvc::Quat& { return self.rotation(); }, [](pytvc::RigidBody& self, const pytvc::Quat& q) {
            self.rotation() = q;
        }, py::return_value_policy::reference_internal)
        .def_property("rotVel", [](pytvc::RigidBody& self) -> pytvc::Vec3& { return self.rot_vel(); }, [](pytvc::RigidBody& self, const pytvc::Vec3& v) {
            self.rot_vel() = v;
        }, py::return_value_policy::reference_internal)
        .def("getAccel", &pytvc::RigidBody::accel)
        .def("applyTorque", &pytvc::RigidBody::apply_torque)
        .def("apply_torque", &pytvc::RigidBody::apply_torque)
        .def("applyLocalTorque", &pytvc::RigidBody::apply_local_torque)
        .def("apply_local_torque", &pytvc::RigidBody::apply_local_torque)
        .def("applyForce", &pytvc::RigidBody::apply_force)
        .def("apply_force", &pytvc::RigidBody::apply_force)
        .def("applyLocalForce", &pytvc::RigidBody::apply_local_force)
        .def("apply_local_force", &pytvc::RigidBody::apply_local_force)
        .def("update", &pytvc::RigidBody::update);

    py::class_<PyMotorCurve>(m, "MotorCurve")
        .def(py::init<>())
        .def("add_point", [](PyMotorCurve& self, pytvc::Scalar time, pytvc::Scalar thrust, pytvc::Scalar mass) {
            throw_on_bad(self.add_point(time, thrust, mass));
        })
        .def("size", &PyMotorCurve::size)
        .def("empty", &PyMotorCurve::empty)
        .def("burnout_time", &PyMotorCurve::burnout_time)
        .def("GetBurnoutTime", &PyMotorCurve::burnout_time)
        .def("thrust_at", &PyMotorCurve::thrust_at)
        .def("GetThrust", &PyMotorCurve::thrust_at)
        .def("mass_at", &PyMotorCurve::mass_at)
        .def("GetMass", &PyMotorCurve::mass_at);

    py::class_<pytvc::Actor, PyActor, std::shared_ptr<pytvc::Actor>>(m, "Actor")
        .def(py::init<>())
        .def("update", &pytvc::Actor::update)
        .def("getForce", &pytvc::Actor::force)
        .def("force", &pytvc::Actor::force)
        .def("getTorque", &pytvc::Actor::torque)
        .def("torque", &pytvc::Actor::torque)
        .def("getMass", &pytvc::Actor::mass)
        .def("mass", &pytvc::Actor::mass);

    py::class_<pytvc::AeroCoefficientModel, PyAeroCoefficientModel, std::shared_ptr<pytvc::AeroCoefficientModel>>(m, "AeroCoefficientModel")
        .def(py::init<>())
        .def("lift_coefficient", &pytvc::AeroCoefficientModel::lift_coefficient)
        .def("drag_coefficient", &pytvc::AeroCoefficientModel::drag_coefficient);

    py::class_<ConstantAero, pytvc::AeroCoefficientModel, std::shared_ptr<ConstantAero>>(m, "ConstantAero")
        .def(py::init<pytvc::Scalar, pytvc::Scalar>(), py::arg("cl"), py::arg("cd"));

    py::class_<pytvc::ServoModel, PyServoModel, std::shared_ptr<pytvc::ServoModel>>(m, "ServoModel")
        .def(py::init<>())
        .def("transfer", &pytvc::ServoModel::transfer);

    py::class_<pytvc::LinkageModel, PyLinkageModel, std::shared_ptr<pytvc::LinkageModel>>(m, "LinkageModel")
        .def(py::init<>())
        .def("linkage", &pytvc::LinkageModel::linkage);

    py::class_<pytvc::PressureModel, PyPressureModel, std::shared_ptr<pytvc::PressureModel>>(m, "PressureModel")
        .def(py::init<>())
        .def("pressure", &pytvc::PressureModel::pressure);

    py::class_<pytvc::ConstantPressure, pytvc::PressureModel, std::shared_ptr<pytvc::ConstantPressure>>(m, "ConstantPressure")
        .def(py::init<pytvc::Scalar>(), py::arg("pressure") = pytvc::kSeaLevelAirDensity);

    py::class_<pytvc::AreaModel, PyAreaModel, std::shared_ptr<pytvc::AreaModel>>(m, "AreaModel")
        .def(py::init<>())
        .def("area", &pytvc::AreaModel::area);

    py::class_<ConstantArea, pytvc::AreaModel, std::shared_ptr<ConstantArea>>(m, "ConstantArea")
        .def(py::init<pytvc::Scalar>(), py::arg("area"));

    py::class_<pytvc::RollDampener, pytvc::Actor, std::shared_ptr<pytvc::RollDampener>>(m, "RollDampener")
        .def(py::init<const pytvc::Vec3&>());

    py::class_<PyMotorMount, pytvc::Actor, std::shared_ptr<PyMotorMount>>(m, "MotorMount")
        .def(py::init<>())
        .def("add_motor", [](PyMotorMount& self, PyMotorCurve& motor) {
            throw_on_bad(self.add_motor(&motor));
        }, py::keep_alive<1, 2>())
        .def("ignite", [](PyMotorMount& self, std::size_t index, pytvc::Scalar time) {
            throw_on_bad(self.ignite(index, time));
        })
        .def("igniteMotor", [](PyMotorMount& self, std::size_t index, pytvc::Scalar time) {
            throw_on_bad(self.ignite(index, time));
        });

    py::class_<PyTVCMount, pytvc::Actor, std::shared_ptr<PyTVCMount>>(m, "TVCMount")
        .def(py::init<>())
        .def(py::init<const pytvc::ServoModel*, const pytvc::LinkageModel*>(),
             py::arg("servo"),
             py::arg("linkage"),
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("set_servo_model", &PyTVCMount::set_servo_model, py::keep_alive<1, 2>())
        .def("set_linkage_model", &PyTVCMount::set_linkage_model, py::keep_alive<1, 2>())
        .def("add_motor", [](PyTVCMount& self, PyMotorCurve& motor) {
            throw_on_bad(self.add_motor(&motor));
        }, py::keep_alive<1, 2>())
        .def("ignite", [](PyTVCMount& self, std::size_t index, pytvc::Scalar time) {
            throw_on_bad(self.ignite(index, time));
        })
        .def("igniteMotor", [](PyTVCMount& self, std::size_t index, pytvc::Scalar time) {
            throw_on_bad(self.ignite(index, time));
        })
        .def("setTargetAngles", &PyTVCMount::set_target_angles)
        .def("set_target_angles", &PyTVCMount::set_target_angles)
        .def("getSetpoint", &PyTVCMount::target_angles)
        .def("target_angles", &PyTVCMount::target_angles)
        .def("getAngles", &PyTVCMount::angles)
        .def("angles", &PyTVCMount::angles);

    py::class_<pytvc::Fin, pytvc::Actor, std::shared_ptr<pytvc::Fin>>(m, "Fin")
        .def(py::init<
                 const pytvc::Vec3&,
                 const pytvc::AeroCoefficientModel*,
                 const pytvc::PressureModel*,
                 pytvc::Scalar,
                 pytvc::Scalar>(),
             py::arg("position"),
             py::arg("coefficients"),
             py::arg("pressure"),
             py::arg("area"),
             py::arg("angle"),
             py::keep_alive<1, 3>(),
             py::keep_alive<1, 4>())
        .def_property_readonly("position", &pytvc::Fin::position)
        .def("setAngle", &pytvc::Fin::set_angle)
        .def("set_angle", &pytvc::Fin::set_angle)
        .def("getAngle", &pytvc::Fin::angle)
        .def("angle", &pytvc::Fin::angle)
        .def("getBodyAngle", &pytvc::Fin::body_angle)
        .def("body_angle", &pytvc::Fin::body_angle)
        .def("getRotationBody", &pytvc::Fin::rotation_body)
        .def("rotation_body", &pytvc::Fin::rotation_body)
        .def("last_lift_body", &pytvc::Fin::last_lift_body)
        .def("last_drag_body", &pytvc::Fin::last_drag_body);

    py::class_<PyFinCan, pytvc::Actor, std::shared_ptr<PyFinCan>>(m, "FinCan")
        .def(py::init<>())
        .def("configure", [](PyFinCan& self,
                             const pytvc::Vec3& position,
                             const pytvc::AeroCoefficientModel* coefficients,
                             const pytvc::PressureModel* pressure,
                             pytvc::Scalar area,
                             pytvc::Scalar radial_distance,
                             std::size_t fin_count,
                             pytvc::Scalar initial_offset) {
            throw_on_bad(self.configure(position, coefficients, pressure, area, radial_distance, fin_count, initial_offset));
        }, py::arg("position"),
           py::arg("coefficients"),
           py::arg("pressure"),
           py::arg("area"),
           py::arg("radial_distance"),
           py::arg("fin_count"),
           py::arg("initial_offset") = 0.0,
           py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>())
        .def("fin_count", &PyFinCan::fin_count)
        .def("getFins", [](PyFinCan& self) {
            py::list fins;
            for (std::size_t i = 0; i < self.fin_count(); ++i) {
                fins.append(py::cast(&self.fin(i), py::return_value_policy::reference_internal));
            }
            return fins;
        })
        .def_property_readonly("position", &PyFinCan::position);

    py::class_<PySpinCan, PyFinCan, std::shared_ptr<PySpinCan>>(m, "SpinCan")
        .def(py::init<>())
        .def("set_roll_coefficient", &PySpinCan::set_roll_coefficient);

    py::class_<pytvc::Airbrake, pytvc::Actor, std::shared_ptr<pytvc::Airbrake>>(m, "Airbrake")
        .def(py::init<
                 const pytvc::Vec3&,
                 pytvc::Scalar,
                 pytvc::Scalar,
                 const pytvc::AeroCoefficientModel*,
                 pytvc::Scalar>(),
             py::arg("position"),
             py::arg("bodyAngle"),
             py::arg("area"),
             py::arg("coefficients"),
             py::arg("maxAngle") = pytvc::Scalar(60) * pytvc::kPi / pytvc::Scalar(180),
             py::keep_alive<1, 5>())
        .def_property_readonly("position", &pytvc::Airbrake::position)
        .def("setAngle", &pytvc::Airbrake::set_angle)
        .def("set_angle", &pytvc::Airbrake::set_angle)
        .def("getAngle", &pytvc::Airbrake::angle)
        .def("angle", &pytvc::Airbrake::angle)
        .def("getBodyAngle", &pytvc::Airbrake::body_angle)
        .def("body_angle", &pytvc::Airbrake::body_angle)
        .def("getLastDrag", &pytvc::Airbrake::last_drag)
        .def("last_drag", &pytvc::Airbrake::last_drag);

    py::class_<PyAirbrakeCan, pytvc::Actor, std::shared_ptr<PyAirbrakeCan>>(m, "AirbrakeCan")
        .def(py::init<>())
        .def("configure", [](PyAirbrakeCan& self,
                             const pytvc::Vec3& position,
                             const pytvc::AeroCoefficientModel* coefficients,
                             pytvc::Scalar area,
                             pytvc::Scalar radial_distance,
                             std::size_t brake_count,
                             pytvc::Scalar initial_offset,
                             pytvc::Scalar max_angle) {
            throw_on_bad(self.configure(position, coefficients, area, radial_distance, brake_count, initial_offset, max_angle));
        }, py::arg("position"),
           py::arg("coefficients"),
           py::arg("area"),
           py::arg("radial_distance"),
           py::arg("brake_count"),
           py::arg("initial_offset") = 0.0,
           py::arg("max_angle") = pytvc::Scalar(60) * pytvc::kPi / pytvc::Scalar(180),
           py::keep_alive<1, 3>())
        .def("setAngle", &PyAirbrakeCan::set_angle)
        .def("set_angle", &PyAirbrakeCan::set_angle)
        .def("brake_count", &PyAirbrakeCan::brake_count)
        .def("getAirbrakes", [](PyAirbrakeCan& self) {
            py::list brakes;
            for (std::size_t i = 0; i < self.brake_count(); ++i) {
                brakes.append(py::cast(&self.brake(i), py::return_value_policy::reference));
            }
            return brakes;
        })
        .def("getDragVectors", [](PyAirbrakeCan& self) {
            py::list drags;
            for (std::size_t i = 0; i < self.brake_count(); ++i) {
                drags.append(self.brake(i).last_drag());
            }
            return drags;
        })
        .def_property_readonly("position", &PyAirbrakeCan::position);

    py::class_<pytvc::RocketBody, pytvc::Actor, std::shared_ptr<pytvc::RocketBody>>(m, "RocketBody")
        .def(py::init<
                 const pytvc::Vec3&,
                 const pytvc::AeroCoefficientModel*,
                 const pytvc::PressureModel*,
                 const pytvc::AreaModel*>(),
             py::arg("position"),
             py::arg("coefficients"),
             py::arg("pressure"),
             py::arg("area"),
             py::keep_alive<1, 3>(),
             py::keep_alive<1, 4>(),
             py::keep_alive<1, 5>())
        .def("getLift", &pytvc::RocketBody::lift)
        .def("lift", &pytvc::RocketBody::lift)
        .def("getDrag", &pytvc::RocketBody::drag)
        .def("drag", &pytvc::RocketBody::drag);

    py::class_<pytvc::SimSample>(m, "SimSample")
        .def_readonly("time", &pytvc::SimSample::time)
        .def_readonly("position", &pytvc::SimSample::position)
        .def_readonly("velocity", &pytvc::SimSample::velocity)
        .def_readonly("accel", &pytvc::SimSample::accel)
        .def_readonly("rotation", &pytvc::SimSample::rotation)
        .def_readonly("rot_vel", &pytvc::SimSample::rot_vel)
        .def_readonly("mass", &pytvc::SimSample::mass);

    py::class_<pytvc::TelemetrySink, std::shared_ptr<pytvc::TelemetrySink>>(m, "TelemetrySink");

    py::class_<PythonTelemetrySink, pytvc::TelemetrySink, std::shared_ptr<PythonTelemetrySink>>(m, "PythonTelemetrySink")
        .def(py::init<py::object>(), py::keep_alive<1, 2>());

    py::class_<PyRocket>(m, "Rocket")
        .def(py::init<
                 pytvc::Scalar,
                 const pytvc::Vec3&,
                 const pytvc::Vec3&,
                 const pytvc::Vec3&,
                 const pytvc::Quat&,
                 const pytvc::Vec3&>(),
             py::arg("dryMass"),
             py::arg("inertia"),
             py::arg("position"),
             py::arg("velocity"),
             py::arg("rotation"),
             py::arg("rotVel"))
        .def("time", &PyRocket::time)
        .def("getDryMass", &PyRocket::dry_mass)
        .def("dry_mass", &PyRocket::dry_mass)
        .def("setDryMass", [](PyRocket& self, pytvc::Scalar dry_mass) {
            throw_on_bad(self.set_dry_mass(dry_mass));
        })
        .def("set_dry_mass", [](PyRocket& self, pytvc::Scalar dry_mass) {
            throw_on_bad(self.set_dry_mass(dry_mass));
        })
        .def("getRigidBody", static_cast<pytvc::RigidBody& (PyRocket::*)()>(&PyRocket::rigid_body), py::return_value_policy::reference_internal)
        .def("rigid_body", static_cast<pytvc::RigidBody& (PyRocket::*)()>(&PyRocket::rigid_body), py::return_value_policy::reference_internal)
        .def("addActor", [](PyRocket& self, std::shared_ptr<pytvc::Actor> actor, const pytvc::Vec3& position) {
            throw_on_bad(self.add_actor(actor.get(), position));
        }, py::arg("actor"),
           py::arg("position") = pytvc::Vec3{},
           py::keep_alive<1, 2>())
        .def("add_actor", [](PyRocket& self, std::shared_ptr<pytvc::Actor> actor, const pytvc::Vec3& position) {
            throw_on_bad(self.add_actor(actor.get(), position));
        }, py::arg("actor"),
           py::arg("position") = pytvc::Vec3{},
           py::keep_alive<1, 2>())
        .def("actor_count", &PyRocket::actor_count)
        .def("update", [](PyRocket& self, pytvc::Scalar dt) {
            throw_on_bad(self.step(dt));
        })
        .def("step", [](PyRocket& self, pytvc::Scalar dt, std::shared_ptr<pytvc::TelemetrySink> telemetry) {
            throw_on_bad(self.step(dt, telemetry.get()));
        }, py::arg("dt"), py::arg("telemetry") = nullptr, py::keep_alive<1, 3>());
}
