#pragma once

#include "pytvc/config.hpp"

namespace pytvc {

inline Scalar scalar_sqrt(Scalar value) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_sqrt(value);
#else
    return __builtin_sqrtf(value);
#endif
}

inline Scalar scalar_sin(Scalar value) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_sin(value);
#else
    return __builtin_sinf(value);
#endif
}

inline Scalar scalar_cos(Scalar value) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_cos(value);
#else
    return __builtin_cosf(value);
#endif
}

inline Scalar scalar_acos(Scalar value) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_acos(value);
#else
    return __builtin_acosf(value);
#endif
}

inline Scalar scalar_asin(Scalar value) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_asin(value);
#else
    return __builtin_asinf(value);
#endif
}

inline Scalar scalar_atan2(Scalar y, Scalar x) {
#if PYTVC_SCALAR_DOUBLE
    return __builtin_atan2(y, x);
#else
    return __builtin_atan2f(y, x);
#endif
}

inline Scalar clamp(Scalar value, Scalar low, Scalar high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

struct Vec3 {
    Scalar x = Scalar(0);
    Scalar y = Scalar(0);
    Scalar z = Scalar(0);

    constexpr Vec3() = default;
    constexpr Vec3(Scalar x_in, Scalar y_in, Scalar z_in) : x(x_in), y(y_in), z(z_in) {}

    constexpr Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    constexpr Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    constexpr Vec3 operator*(const Vec3& other) const {
        return {x * other.x, y * other.y, z * other.z};
    }

    constexpr Vec3 operator*(Scalar scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    constexpr Vec3 operator/(const Vec3& other) const {
        return {x / other.x, y / other.y, z / other.z};
    }

    constexpr Vec3 operator/(Scalar scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3& zero() {
        x = Scalar(0);
        y = Scalar(0);
        z = Scalar(0);
        return *this;
    }

    Vec3& add_scaled(const Vec3& other, Scalar scale) {
        x += other.x * scale;
        y += other.y * scale;
        z += other.z * scale;
        return *this;
    }

    constexpr Scalar mag2() const {
        return x * x + y * y + z * z;
    }

    Scalar len() const {
        return scalar_sqrt(mag2());
    }

    Vec3 normalized() const {
        const Scalar m2 = mag2();
        if (m2 == Scalar(0)) {
            return {};
        }
        const Scalar inv = Scalar(1) / scalar_sqrt(m2);
        return {x * inv, y * inv, z * inv};
    }

    constexpr Scalar dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    constexpr Vec3 cross(const Vec3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x,
        };
    }

    Scalar angle_between(const Vec3& other) const {
        const Scalar a2 = mag2();
        const Scalar b2 = other.mag2();
        if (a2 == Scalar(0) || b2 == Scalar(0)) {
            return Scalar(0);
        }
        const Scalar c = clamp(dot(other) / scalar_sqrt(a2 * b2), Scalar(-1), Scalar(1));
        return scalar_acos(c);
    }
};

constexpr Vec3 operator*(Scalar scalar, const Vec3& vec) {
    return vec * scalar;
}

struct Quat {
    Scalar w = Scalar(1);
    Scalar x = Scalar(0);
    Scalar y = Scalar(0);
    Scalar z = Scalar(0);

    constexpr Quat() = default;
    constexpr Quat(Scalar w_in, Scalar x_in, Scalar y_in, Scalar z_in)
        : w(w_in), x(x_in), y(y_in), z(z_in) {}

    constexpr Quat operator+(const Quat& other) const {
        return {w + other.w, x + other.x, y + other.y, z + other.z};
    }

    constexpr Quat operator-(const Quat& other) const {
        return {w - other.w, x - other.x, y - other.y, z - other.z};
    }

    constexpr Quat operator*(Scalar scalar) const {
        return {w * scalar, x * scalar, y * scalar, z * scalar};
    }

    constexpr Quat operator*(const Quat& other) const {
        return {
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w,
        };
    }

    constexpr Quat operator/(Scalar scalar) const {
        if (scalar == Scalar(0)) {
            return {Scalar(0), Scalar(0), Scalar(0), Scalar(0)};
        }
        return {w / scalar, x / scalar, y / scalar, z / scalar};
    }

    constexpr Quat conjugate() const {
        return {w, -x, -y, -z};
    }

    constexpr Vec3 xyz() const {
        return {x, y, z};
    }

    constexpr Scalar mag2() const {
        return w * w + x * x + y * y + z * z;
    }

    Scalar len() const {
        return scalar_sqrt(mag2());
    }

    Quat normalized() const {
        const Scalar m2 = mag2();
        if (m2 == Scalar(0)) {
            return {};
        }
        const Scalar inv = Scalar(1) / scalar_sqrt(m2);
        return {w * inv, x * inv, y * inv, z * inv};
    }

    Quat& normalize_in_place() {
        *this = normalized();
        return *this;
    }

    constexpr Scalar dot(const Quat& other) const {
        return w * other.w + x * other.x + y * other.y + z * other.z;
    }

    Vec3 rotate(const Vec3& v) const {
        const Scalar tx = Scalar(2) * (y * v.z - z * v.y);
        const Scalar ty = Scalar(2) * (z * v.x - x * v.z);
        const Scalar tz = Scalar(2) * (x * v.y - y * v.x);
        return {
            v.x + w * tx + (y * tz - z * ty),
            v.y + w * ty + (z * tx - x * tz),
            v.z + w * tz + (x * ty - y * tx),
        };
    }

    Vec3 rotate_safe(const Vec3& v) const {
        return normalized().rotate(v);
    }

    static Quat from_axis_angle(const Vec3& unit_axis, Scalar angle) {
        const Scalar half = Scalar(0.5) * angle;
        const Scalar s = scalar_sin(half);
        const Scalar c = scalar_cos(half);
        return {c, unit_axis.x * s, unit_axis.y * s, unit_axis.z * s};
    }

    static Quat from_euler(const Vec3& rot) {
        const Scalar hx = Scalar(0.5) * rot.x;
        const Scalar hy = Scalar(0.5) * rot.y;
        const Scalar hz = Scalar(0.5) * rot.z;
        const Scalar cr = scalar_cos(hx);
        const Scalar sr = scalar_sin(hx);
        const Scalar cp = scalar_cos(hy);
        const Scalar sp = scalar_sin(hy);
        const Scalar cy = scalar_cos(hz);
        const Scalar sy = scalar_sin(hz);
        return {
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        };
    }

    Vec3 to_euler() const {
        const Quat q = normalized();
        const Scalar sinr_cosp = Scalar(2) * (q.w * q.x + q.y * q.z);
        const Scalar cosr_cosp = Scalar(1) - Scalar(2) * (q.x * q.x + q.y * q.y);
        const Scalar roll = scalar_atan2(sinr_cosp, cosr_cosp);

        Scalar sinp = Scalar(2) * (q.w * q.y - q.z * q.x);
        sinp = clamp(sinp, Scalar(-1), Scalar(1));
        const Scalar pitch = scalar_asin(sinp);

        const Scalar siny_cosp = Scalar(2) * (q.w * q.z + q.x * q.y);
        const Scalar cosy_cosp = Scalar(1) - Scalar(2) * (q.y * q.y + q.z * q.z);
        const Scalar yaw = scalar_atan2(siny_cosp, cosy_cosp);
        return {roll, pitch, yaw};
    }
};

constexpr Quat operator*(Scalar scalar, const Quat& quat) {
    return quat * scalar;
}

}  // namespace pytvc
