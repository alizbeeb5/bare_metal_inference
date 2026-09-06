#pragma once
#include "arena.hpp"
#include <cstddef>

namespace bmie {
    inline constexpr std::size_t kMaxFeatures =64;


    struct Model {
        std::size_t n_features{};
        float* mean = nullptr;
        float* stddev = nullptr;
        float* weights = nullptr;
        float bias = 0.0f;
    };

    Model* load_model(Arena& arena, const char* path) noexcept;

    float predict(const Model& m, const float* features) noexcept;

    std::size_t model_footprint(const Model& m) noexcept;


}