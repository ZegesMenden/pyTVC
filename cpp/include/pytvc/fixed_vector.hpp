#pragma once

#include <array>
#include <cstddef>

#include "pytvc/config.hpp"

namespace pytvc {

template <typename T, std::size_t Capacity>
class FixedVector {
public:
    using value_type = T;

    constexpr std::size_t capacity() const {
        return Capacity;
    }

    constexpr std::size_t size() const {
        return size_;
    }

    constexpr bool empty() const {
        return size_ == 0;
    }

    constexpr bool full() const {
        return size_ >= Capacity;
    }

    Status push_back(const T& value) {
        if (full()) {
            return Status::full;
        }
        data_[size_] = value;
        ++size_;
        return Status::ok;
    }

    Status pop_back() {
        if (empty()) {
            return Status::empty;
        }
        --size_;
        return Status::ok;
    }

    void clear() {
        size_ = 0;
    }

    T& operator[](std::size_t index) {
        return data_[index];
    }

    const T& operator[](std::size_t index) const {
        return data_[index];
    }

    T* begin() {
        return data_.data();
    }

    T* end() {
        return data_.data() + size_;
    }

    const T* begin() const {
        return data_.data();
    }

    const T* end() const {
        return data_.data() + size_;
    }

private:
    std::array<T, Capacity> data_{};
    std::size_t size_ = 0;
};

}  // namespace pytvc
