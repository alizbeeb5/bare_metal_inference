#include "model.hpp"
#include "math.hpp"
#include <cstdio>
#include <cstring>

namespace bmie {
    namespace {
        constexpr std::uint32_t kVersion = 1;
        constexpr std::size_t kHeaderBytes = 16;
    }
    Model* load_model(Arena& arena, const char* path) noexcept {
        FILE* fp = std::fopen(path, "rb");
        if (!fp) {
            std::fprintf(stderr, "[model] Failed to open %s\n", path);
            return nullptr;
        }

        std::uint8_t header[kHeaderBytes];
        if(std::fread(header, 1, kHeaderBytes, fp) != kHeaderBytes) {
            std::fprintf(stderr, "[model] short read on header\n");
            std::fclose(fp);
            return nullptr;
        }

        if (std::memcmp(header, "BMIE", 4)!=0) {
            std::fprintf(stderr, "[model] bad magic\n");
            std::fclose(fp);
            return nullptr;
        }

        auto read_u32_le = [](const std::uint8_t* p){
            return static_cast<std::uint32_t>(p[0])|
            (static_cast<std::uint32_t>(p[1])<<8) |
            (static_cast<std::uint32_t>(p[2])<<16) |
            (static_cast<std::uint32_t>(p[3])<<24);
        };

        const std::uint32_t version = read_u32_le(header+4);
        const std::uint32_t n_features= read_u32_le(header+8);

        if (version != kVersion) {
            std::fprintf(stderr, "[model] unsupported version\n");
            std::fclose(fp);
            return nullptr;
        }

        if (n_features == 0 || n_features > kMaxFeatures) {
            std::fprintf(stderr, "[model] features are not supported\n");
            std::fclose(fp);
            return nullptr;
        }


        const std::size_t body_bytes = (3*n_features+1)*sizeof(float);

        if (arena.remaining() < body_bytes) {
            std::fprintf(stderr, "[model] arena budget is too small %zu, need %zu\n", arena.remaining(), body_bytes);
            std::fclose(fp);
            return nullptr;
        }

        float* mean = arena.alloc<float>(n_features);
        float* weights = arena.alloc<float>(n_features);
        float* stddev = arena.alloc<float>(n_features);


        if (!mean || !stddev || !weights) {
            std::fprintf(stderr, "[model] arena alloc failed\n");
            std::fclose(fp);
            return nullptr;
        }

        if (std::fread(mean, sizeof(float), n_features, fp) != n_features ||
            std::fread(stddev, sizeof(float), n_features, fp) != n_features||
            std::fread(weights, sizeof(float), n_features, fp) != n_features
             ) {

            std::fprintf(stderr, "[model] short read on body\n");
            std::fclose(fp);
            return nullptr;
        }

        float bias = 0.0f;
        if (std::fread(&bias, sizeof(float), 1, fp) != 1) {
            std::fprintf(stderr, "[model] short read on bias\n");
            std::fclose(fp);
            return nullptr;
        }
        std::fclose(fp);

        Model* m = arena.alloc<Model>(1);
        if (!m) {
            std::fprintf(stderr, "[model] arena alloc for Model failed\n");
            return nullptr;
        }

        m->n_features = n_features;
        m->mean = mean;
        m->stddev = stddev;
        m->weights = weights;
        m->bias = bias;
        return m;

    }

    float predict(const Model& m ,const float* features) noexcept {
        float x[kMaxFeatures];
        for (std::size_t i =0 ; i<m.n_features; ++i) {
            x[i] = features[i];
        }

        math::normalize(x, m.mean, m.stddev, m.n_features);

        float z = 0.0f;
        math::gemv(&z, m.weights, x, m.bias, 1, m.n_features);

        return math::sigmoid(z);
    }

    std::size_t model_footprint(const Model& m) noexcept {
        return (3*m.n_features+1)*sizeof(float) + sizeof(Model);
    }

}