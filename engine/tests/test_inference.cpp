// ctest --test-dir engine/build --output-on-failure

#include "arena.hpp"
#include "math.hpp"
#include "model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;


#define EXPECT(cond) do {                                              \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "FAIL: %s @ %s:%d\n",                     \
                     #cond, __FILE__, __LINE__);                       \
        std::exit(1);                                                  \
    }                                                                  \
} while (0)

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


struct Reference {
    std::vector<float> mean;
    std::vector<float> stddev;
    std::vector<float> weights;
    float             bias = 0.0f;
};

bool load_reference(const char* scaler_path,
                    const char* weights_path,
                    Reference& r) {
    std::ifstream f(scaler_path);
    if (!f) {
        std::fprintf(stderr, "[ref] cannot open %s\n", scaler_path);
        return false;
    }
    json j;
    try { f >> j; } catch (const std::exception& e) {
        std::fprintf(stderr, "[ref] parse %s failed: %s\n", scaler_path, e.what());
        return false;
    }
    r.mean   = j.at("mean").get<std::vector<float>>();
    r.stddev = j.at("std").get<std::vector<float>>();

    std::ifstream fw(weights_path);
    if (!fw) {
        std::fprintf(stderr, "[ref] cannot open %s\n", weights_path);
        return false;
    }
    json jw;
    try { fw >> jw; } catch (const std::exception& e) {
        std::fprintf(stderr, "[ref] parse %s failed: %s\n", weights_path, e.what());
        return false;
    }
    r.weights = jw.at("weights").get<std::vector<float>>();
    r.bias    = jw.at("bias").get<float>();
    return true;
}

float reference_predict(const Reference& r, const float* features) {
    constexpr std::size_t kMax = 64;
    float x[kMax];
    for (std::size_t i = 0; i < r.weights.size(); ++i) x[i] = features[i];

    bmie::math::normalize(x, r.mean.data(), r.stddev.data(), r.weights.size());

    float z = 0.0f;
    bmie::math::gemv(&z, r.weights.data(), x, r.bias, 1, r.weights.size());

    return bmie::math::sigmoid(z);
}



int main(int argc, char** argv) {

    const char* model_path        = (argc > 1) ? argv[1] : "model.bin";
    const char* scaler_json_path  = (argc > 2) ? argv[2] : "trainer/data/scaler.json";
    const char* weights_json_path = (argc > 3) ? argv[3] : "trainer/data/weights.json";


    bmie::Arena arena(64 * 1024);  // 64 KB budget, plenty for our 8-feature model
    bmie::Model* model = bmie::load_model(arena, model_path);
    EXPECT(model != nullptr);
    EXPECT(model->n_features == 8);


    Reference ref;
    EXPECT(load_reference(scaler_json_path, weights_json_path, ref));
    EXPECT(ref.weights.size() == 8);
    EXPECT(ref.mean.size() == 8);
    EXPECT(ref.stddev.size() == 8);


    const std::vector<std::vector<float>> cases = {
        // lead_time, reliability, distance, season, order_value, item_cat, carrier, weekend
        { 5.0f, 0.95f,  500.0f, 0.0f,  100.0f, 0.0f, 0.0f, 0.0f },  // trivial
        {25.0f, 0.55f, 4500.0f, 3.0f, 4000.0f, 4.0f, 2.0f, 1.0f },  // pathological
        {15.0f, 0.75f, 2500.0f, 1.0f, 2500.0f, 2.0f, 1.0f, 0.0f },  // typical
        { 1.0f, 1.00f,   50.0f, 0.0f,   10.0f, 0.0f, 0.0f, 0.0f },  // very easy
        {30.0f, 0.50f, 5000.0f, 3.0f, 5000.0f, 4.0f, 2.0f, 1.0f },  // very hard
    };

    int pass = 0, fail = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const float p_cpp  = bmie::predict(*model, cases[i].data());
        const float p_ref  = reference_predict(ref, cases[i].data());
        const double diff  = std::fabs(static_cast<double>(p_cpp) -
                                       static_cast<double>(p_ref));

        std::printf("case %zu: cpp=%.6f  ref=%.6f  diff=%.2e",
                    i, p_cpp, p_ref, diff);

        if (diff < 1e-5) {
            std::printf("  OK\n");
            ++pass;
        } else {
            std::printf("  FAIL\n");
            ++fail;
        }
    }

    EXPECT(fail == 0);
    std::printf("test_inference: %d/%d cases matched\n", pass, (int)cases.size());


    const std::size_t fp = bmie::model_footprint(*model);
    std::printf("model footprint: %zu bytes (%.1f%% of 64 KB)\n",
                fp, 100.0 * fp / (64.0 * 1024));
    EXPECT(fp < 64 * 1024);


    const std::size_t used_before = arena.used();
    for (int i = 0; i < 1000; ++i) {
        volatile float p = bmie::predict(*model, cases[0].data());
        (void)p;
    }
    const std::size_t used_after = arena.used();
    std::printf("arena used: %zu before, %zu after 1000 calls\n",
                used_before, used_after);
    EXPECT(used_after == used_before);  // no growth = no allocations

    std::printf("test_inference: all tests passed\n");
    return 0;
}