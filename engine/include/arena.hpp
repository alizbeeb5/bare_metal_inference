#pragma once
#include <cstddef>
#include <cstdint>
#include <new>

namespace bmie {
    class Arena {
    public:
        explicit Arena(std::size_t capacity_bytes)
            : base_(new std::byte[capacity_bytes]),
              capacity_(capacity_bytes),
              offset_(0),
              oom_(false) {}


        ~Arena() {
            delete[] base_;
        }

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;
        Arena(Arena&&) = delete;
        Arena& operator=(Arena&&) = delete;
        template <typename T>
        T* alloc(std::size_t count = 1) noexcept {
            const std::size_t bytes = count * sizeof(T);
            const std::size_t aligned_offset = align_up(offset_, alignof(T));
            if (aligned_offset + bytes > capacity_) {
                oom_ = true;
                return nullptr;
            }
            T* ptr = std::launder(reinterpret_cast<T*>(base_ + aligned_offset));
            offset_ = aligned_offset + bytes;
            return ptr;
        }

        std::byte* alloc_bytes(std::size_t bytes, std::size_t alignment) noexcept {
            const std::size_t aligned_offset = align_up(offset_, alignment);
            if (aligned_offset + bytes > capacity_) {
                oom_ = true;
                return nullptr;
            }
            std::byte* ptr = base_ +aligned_offset;
            offset_ = aligned_offset +bytes;
            return ptr;
        }

        void reset() noexcept {
            offset_ = 0;
            oom_ = false;

        }

        std::size_t capacity() const noexcept {return capacity_;}
        std::size_t used() const noexcept {return offset_;}
        std:: size_t remaining() const noexcept {return capacity_ - offset_;}
        bool         oom()       const noexcept {return oom_;}
    private:
        static std::size_t align_up(std::size_t n, std::size_t alignment) noexcept {
            return (n + alignment - 1) & ~(alignment - 1);
        }
        std::byte* base_;
        std::size_t capacity_;
        std::size_t offset_;
        bool oom_;

    };
}