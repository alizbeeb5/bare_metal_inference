#pragma once
#include <cmath>
#include <cstdlib>

namespace bmie::math {
    inline void gemv(float* __restrict__ y,
        const float* __restrict__ A,
        const float* __restrict__ x,
        const float bias,
        const std::size_t m,
        const std::size_t n) noexcept {
        for (std::size_t i = 0; i < m; ++i) {
            const float* row = A +i*n;
            float acc = bias;
            for (std::size_t j = 0; j < n; ++j) {
                acc += x[j]*row[j];
            }
            y[i] = acc;
        }
    }


    inline float sigmoid(const float z) noexcept {
        if (z >= 0.0f) {
            const float e = std::exp(-z);
            return 1.0f / (1.0f + e);
        }
        else {
            const float e = std::exp(z);
            return e / (1.0f + e);
        }
    }

    inline void normalize(float* __restrict__ x ,
        const float* __restrict__  mean,
        const float* __restrict__  stddev,
        std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) {
            const float s = (stddev[i]!=0.0f) ? stddev[i] : 1.0f;
            x[i] = (x[i] - mean[i]) / s;
        }



    }
}
