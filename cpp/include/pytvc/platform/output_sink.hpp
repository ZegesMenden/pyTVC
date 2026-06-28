#pragma once

#include <cstddef>

namespace pytvc {

class OutputSink {
public:
    virtual ~OutputSink() = default;
    virtual void write(const char* data, std::size_t length) = 0;
};

class NullOutputSink final : public OutputSink {
public:
    void write(const char*, std::size_t) override {}
};

template <std::size_t Capacity>
class BufferOutputSink final : public OutputSink {
public:
    static_assert(Capacity > 0, "BufferOutputSink capacity must reserve at least one byte");

    void write(const char* data, std::size_t length) override {
        for (std::size_t i = 0; i < length && size_ + 1 < Capacity; ++i) {
            buffer_[size_] = data[i];
            ++size_;
        }
        if (Capacity > 0) {
            buffer_[size_] = '\0';
        }
    }

    const char* c_str() const {
        return buffer_;
    }

    std::size_t size() const {
        return size_;
    }

    void clear() {
        size_ = 0;
        if (Capacity > 0) {
            buffer_[0] = '\0';
        }
    }

private:
    char buffer_[Capacity]{};
    std::size_t size_ = 0;
};

}  // namespace pytvc
