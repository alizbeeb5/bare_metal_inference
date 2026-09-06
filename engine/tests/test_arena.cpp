// test_arena.cpp — unit tests for the hand-rolled arena allocator.
//
// Run via: ctest --test-dir engine/build --output-on-failure
// Or directly: ./test_arena

#include "arena.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using bmie::Arena;

#define EXPECT(cond) do {                                              \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n",                     \
                     #cond, __FILE__, __LINE__);                       \
        std::exit(1);                                                  \
    }                                                                  \
} while (0)

// --- Test 1: basic allocation is usable ---
//
// The most important property: alloc returns a pointer you can actually
// write to and read from. If this fails, every other test is meaningless.
void test_basic_alloc() {
    Arena a(1024);
    int* p = a.alloc<int>(4);                  // 16 bytes
    EXPECT(p != nullptr);
    EXPECT(a.used() == 16);
    EXPECT(a.remaining() == 1008);
    EXPECT(!a.oom());

    for (int i = 0; i < 4; ++i) p[i] = i * 10;
    EXPECT(p[0] == 0);
    EXPECT(p[1] == 10);
    EXPECT(p[2] == 20);
    EXPECT(p[3] == 30);
}

// --- Test 2: alignment of typed allocations ---
//
// A uint8_t is 1-byte aligned, a uint32_t is 4-byte aligned. After
// allocating 1 byte, the next uint32_t must still land on a 4-byte
// boundary. If align_up() is wrong, the pointer is misaligned and
// dereferencing it is UB on strict-alignment architectures (ARM, etc).
void test_alignment() {
    Arena a(1024);

    std::uint8_t* p1 = a.alloc<std::uint8_t>(1);   // offset 0..1
    EXPECT(reinterpret_cast<std::uintptr_t>(p1) % 1 == 0);

    std::uint32_t* p2 = a.alloc<std::uint32_t>(1); // offset bumped to 4
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p2);
    EXPECT(addr % alignof(std::uint32_t) == 0);

    // Worst case: 7 bytes consumed, then ask for 8-byte alignment.
    a.reset();
    a.alloc<std::uint8_t>(7);                      // offset = 7
    double* d = a.alloc<double>(1);                // must be 8-aligned
    EXPECT(reinterpret_cast<std::uintptr_t>(d) % alignof(double) == 0);
}

// --- Test 3: exhaustion path returns nullptr and sets oom_ ---
//
// The arena must refuse to allocate past its budget. The right
// behavior is nullptr + sticky oom_ flag, not a crash, not a throw.
void test_exhaustion() {
    Arena a(64);
    float* p = a.alloc<float>(16);                 // exactly fills 64 bytes
    EXPECT(p != nullptr);
    EXPECT(a.remaining() == 0);
    EXPECT(!a.oom());                              // hasn't failed yet

    float* q = a.alloc<float>(1);                  // one byte too many
    EXPECT(q == nullptr);
    EXPECT(a.oom());                               // flag is now set

    // oom_ is sticky: even a tiny subsequent alloc fails.
    EXPECT(a.alloc<int>(1) == nullptr);
    EXPECT(a.oom());
}

// --- Test 4: reset() restores the arena to a fresh state ---
//
// After reset, the bump pointer is back to 0, oom_ is cleared, and
// allocations work again. This is the O(1) "free everything" promise.
void test_reset() {
    Arena a(256);
    a.alloc<float>(32);                            // 128 bytes used
    EXPECT(a.used() == 128);

    a.reset();
    EXPECT(a.used() == 0);
    EXPECT(a.remaining() == 256);
    EXPECT(!a.oom());

    // After reset, the same allocation must succeed again.
    float* p = a.alloc<float>(64);
    EXPECT(p != nullptr);

    // Reset also clears a prior oom flag.
    a.reset();
    Arena b(16);
    (void)b.alloc<int>(100);                       // force oom
    EXPECT(b.oom());
    b.reset();
    EXPECT(!b.oom());
    int* p2 = b.alloc<int>(1);
    EXPECT(p2 != nullptr);
}

// --- Test 5: many small allocs, all unique and aligned ---
//
// 1000 single-byte allocations. Every returned pointer must be
// 4-byte aligned (because int) and unique. Catches a class of bugs
// where the bump pointer drifts or aliases.
void test_many_small_allocs() {
    Arena a(4096);
    std::vector<int*> ptrs;
    for (int i = 0; i < 1000; ++i) {
        int* p = a.alloc<int>(1);
        EXPECT(p != nullptr);
        ptrs.push_back(p);
    }

    // Every pointer must be aligned and unique.
    for (std::size_t i = 0; i < ptrs.size(); ++i) {
        EXPECT(reinterpret_cast<std::uintptr_t>(ptrs[i]) % alignof(int) == 0);
        for (std::size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT(ptrs[i] != ptrs[j]);
        }
    }
}

// --- Test 6: alloc_bytes — would have caught the + vs = bug ---
void test_alloc_bytes() {
    Arena a(256);
    for (int i = 0; i < 8; ++i) {
        std::byte* p = a.alloc_bytes(8, 4);
        EXPECT(p != nullptr);
        // Each pointer must be 4-byte aligned.
        EXPECT(reinterpret_cast<std::uintptr_t>(p) % 4 == 0);
    }
    EXPECT(a.used() == 64);
    EXPECT(a.remaining() == 192);
    EXPECT(!a.oom());

    // Mixed alloc<T> and alloc_bytes in the same arena.
    a.reset();
    int* pi = a.alloc<int>(2);                     // 8 bytes
    EXPECT(pi != nullptr);
    std::byte* pb = a.alloc_bytes(16, 8);          // 16 bytes
    EXPECT(pb != nullptr);
    EXPECT(reinterpret_cast<std::uintptr_t>(pb) % 8 == 0);
    EXPECT(a.used() == 24);

    // Exhaustion on alloc_bytes works the same way.
    Arena small(16);
    std::byte* ok = small.alloc_bytes(16, 1);
    EXPECT(ok != nullptr);
    std::byte* fail = small.alloc_bytes(1, 1);
    EXPECT(fail == nullptr);
    EXPECT(small.oom());
}

int main() {
    test_basic_alloc();
    test_alignment();
    test_exhaustion();
    test_reset();
    test_many_small_allocs();
    test_alloc_bytes();
    std::printf("test_arena: all tests passed\n");
    return 0;
}