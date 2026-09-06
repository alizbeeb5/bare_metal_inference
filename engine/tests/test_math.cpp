// test_math.cpp — unit tests for the math primitives.
//
// Run via: ctest --test-dir engine/build --output-on-failure
// Or directly: ./test_math

#include "math.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

using bmie::math::gemv;
using bmie::math::sigmoid;
using bmie::math::normalize;

// EXPECT(cond): exits with code 1 on failure, prints the failing
// expression and source location. Same pattern as test_arena.cpp.
#define EXPECT(cond) do {                                              \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n",                     \
                     #cond, __FILE__, __LINE__);                       \
        std::exit(1);                                                  \
    }                                                                  \
} while (0)

// EXPECT_NEAR(a, b, eps): like EXPECT but for floating-point.
// 1e-5 is tight enough for float32 to catch real bugs, loose enough
// to not flake on rounding noise.
#define EXPECT_NEAR(a, b, eps) do {                                            \
    double _a = static_cast<double>(a);                                        \
    double _b = static_cast<double>(b);                                        \
    if (std::fabs(_a - _b) > (eps)) {                                          \
        std::fprintf(stderr, "FAIL: %s=%g vs %s=%g, diff=%g @ %s:%d\n",        \
                     #a, _a, #b, _b, std::fabs(_a - _b),                       \
                     __FILE__, __LINE__);                                      \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// =====================================================================
// gemv: y = A @ x + bias  (A is m x n, x is n, y is m)
// =====================================================================

// The most basic test: a known small matrix multiplied by a known
// vector gives a known answer. If the loop bounds are off, the dot
// products will be wrong.
void test_gemv_basic() {
    // A = [1 2 3
    //      4 5 6]
    // x = [1, 1, 1]
    // y_expected = [1+2+3, 4+5+6] = [6, 15]
    float A[6] = {1, 2, 3,
                  4, 5, 6};
    float x[3] = {1, 1, 1};
    float y[2] = {0, 0};

    gemv(y, A, x, 0.0f, 2, 3);

    EXPECT_NEAR(y[0], 6.0f,  1e-6);
    EXPECT_NEAR(y[1], 15.0f, 1e-6);
}

// The bias is added to every output element. Without it, the
// "model with bias" case gives wrong results.
void test_gemv_with_bias() {
    // A = [2, 3], x = [1, 1], bias = 10
    // y = 2*1 + 3*1 + 10 = 15
    float A[2] = {2, 3};
    float x[2] = {1, 1};
    float y[1] = {0};

    gemv(y, A, x, 10.0f, 1, 2);
    EXPECT_NEAR(y[0], 15.0f, 1e-6);
}

// If A is all zeros, the output is just bias broadcast across m rows.
void test_gemv_zeros() {
    float A[3] = {0, 0, 0};
    float x[3] = {5, 6, 7};  // any values, they get multiplied by zero
    float y[1] = {0};

    gemv(y, A, x, 0.0f, 1, 3);
    EXPECT_NEAR(y[0], 0.0f, 1e-6);

    // Same thing but with bias.
    y[0] = 0;
    gemv(y, A, x, -2.5f, 1, 3);
    EXPECT_NEAR(y[0], -2.5f, 1e-6);
}

// The case we'll actually use at inference: m=1 (one output neuron).
// The forward pass is effectively a dot product + bias.
void test_gemv_single_output() {
    // A = [0.6, -0.5, 0.57, 0.15, 0.14, -0.19, -0.16, 0.34]  (8 weights)
    // x = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]      (8 normalized features)
    // dot = 0.6*0.1 + (-0.5)*(-0.2) + 0.57*0.3 + ...        (compute by hand below)
    // bias = -0.92
    float w[8] = {0.6f, -0.5f, 0.57f, 0.15f, 0.14f, -0.19f, -0.16f, 0.34f};
    float x[8] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
    float y[1] = {0};

    gemv(y, w, x, -0.92f, 1, 8);

    // dot = 0.06 + 0.10 + 0.171 - 0.06 + 0.07 + 0.114 - 0.112 - 0.272
    //     = 0.061 (approximately)
    // y = 0.061 + (-0.92) = -0.859
    // The exact value doesn't matter what matters is that gemv
    // gives a reproducible answer that we can sanity-check against
    // a hand calculation.
    const float dot = 0.06f + 0.10f + 0.171f - 0.06f + 0.07f + 0.114f - 0.112f - 0.272f;
    EXPECT_NEAR(y[0], dot - 0.92f, 1e-5);
}

// =====================================================================
// sigmoid: branch-stable, [0, 1] output
// =====================================================================

// sigmoid(0) = 0.5 exactly. If this fails, the if/else is busted.
void test_sigmoid_zero() {
    EXPECT_NEAR(sigmoid(0.0f), 0.5f, 1e-6);
}

// Large positive -> 1.0. Catches the if-branch.
void test_sigmoid_large_positive() {
    EXPECT_NEAR(sigmoid(20.0f), 1.0f, 1e-5);
    EXPECT_NEAR(sigmoid(5.0f),  0.9933f, 1e-3);
}

// Large negative -> 0.0. This is the test that catches the
// else-branch bug we just fixed (1/(1+e) returns ~0.73 instead of ~0).
void test_sigmoid_large_negative() {
    EXPECT_NEAR(sigmoid(-20.0f), 0.0f,  1e-5);
    EXPECT_NEAR(sigmoid(-5.0f),  0.0067f, 1e-3);
}

// Stability: must not produce NaN or Inf at extreme inputs.
// This is why the branch-stable version exists.
void test_sigmoid_stability() {
    EXPECT(std::isfinite(sigmoid( 1000.0f)));
    EXPECT(std::isfinite(sigmoid(-1000.0f)));
    EXPECT(std::isfinite(sigmoid( 100.0f)));
    EXPECT(std::isfinite(sigmoid(-100.0f)));
    // Saturation: the answers must be very close to 0 or 1, not garbage.
    EXPECT(sigmoid( 1000.0f) > 0.9999f);
    EXPECT(sigmoid(-1000.0f) < 0.0001f);
}

// =====================================================================
// normalize: x[i] = (x[i] - mean[i]) / stddev[i], in-place
// =====================================================================

// Standard case: known input, known mean/std, known output.
void test_normalize_basic() {
    float x[3]   = {10.0f, 20.0f, 30.0f};
    float mean[3] = {10.0f, 20.0f, 30.0f};
    float std[3]  = {2.0f,   5.0f, 10.0f};

    normalize(x, mean, std, 3);

    // (10-10)/2 = 0
    // (20-20)/5 = 0
    // (30-30)/10 = 0
    EXPECT_NEAR(x[0], 0.0f, 1e-6);
    EXPECT_NEAR(x[1], 0.0f, 1e-6);
    EXPECT_NEAR(x[2], 0.0f, 1e-6);
}

// Non-trivial values: known (x, mean, std) -> known normalized.
void test_normalize_nonzero() {
    float x[2]   = {12.0f, 25.0f};
    float mean[2] = {10.0f, 20.0f};
    float std[2]  = {2.0f,   5.0f};

    normalize(x, mean, std, 2);

    // (12-10)/2 = 1
    // (25-20)/5 = 1
    EXPECT_NEAR(x[0], 1.0f, 1e-6);
    EXPECT_NEAR(x[1], 1.0f, 1e-6);
}

// stddev[i] == 0 must not crash or produce NaN. The function should
// fall back to dividing by 1, making that feature contribute 0.
void test_normalize_zero_std() {
    float x[2]   = {5.0f, 100.0f};
    float mean[2] = {5.0f, 50.0f};
    float std[2]  = {0.0f, 0.0f};

    normalize(x, mean, std, 2);

    // x[0] = (5-5)/1   = 0
    // x[1] = (100-50)/1 = 50
    EXPECT_NEAR(x[0], 0.0f,  1e-6);
    EXPECT_NEAR(x[1], 50.0f, 1e-6);
    EXPECT(!std::isnan(x[0]));
    EXPECT(!std::isnan(x[1]));
}

// Mixed: some real std, some zero std. Common in practice.
void test_normalize_mixed_std() {
    float x[4]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float mean[4] = {1.0f, 0.0f, 3.0f, 0.0f};
    float std[4]  = {1.0f, 2.0f, 0.0f, 1.0f};

    normalize(x, mean, std, 4);

    EXPECT_NEAR(x[0], 0.0f, 1e-6);   // (1-1)/1
    EXPECT_NEAR(x[1], 1.0f, 1e-6);   // (2-0)/2
    EXPECT_NEAR(x[2], 0.0f, 1e-6);   // (3-3)/1 (fallback)
    EXPECT_NEAR(x[3], 4.0f, 1e-6);   // (4-0)/1
}

// In-place confirmation: the function must modify x, not return a copy.
void test_normalize_in_place() {
    float x[1]   = {99.0f};
    float mean[1] = {0.0f};
    float std[1]  = {10.0f};

    // Capture the address before the call.
    float* before = x;
    normalize(x, mean, std, 1);
    float* after = x;

    EXPECT(before == after);
    EXPECT_NEAR(x[0], 9.9f, 1e-6);
}

int main() {
    // gemv
    test_gemv_basic();
    test_gemv_with_bias();
    test_gemv_zeros();
    test_gemv_single_output();

    // sigmoid
    test_sigmoid_zero();
    test_sigmoid_large_positive();
    test_sigmoid_large_negative();
    test_sigmoid_stability();

    // normalize
    test_normalize_basic();
    test_normalize_nonzero();
    test_normalize_zero_std();
    test_normalize_mixed_std();
    test_normalize_in_place();

    std::printf("test_math: all tests passed\n");
    return 0;
}