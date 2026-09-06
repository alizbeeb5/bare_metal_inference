#include "api.hpp"
#include "arena.hpp"
#include "model.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {

    const char* model_path = (argc > 1) ? argv[1] : "model.bin";
    const char* host       = (argc > 2) ? argv[2] : "0.0.0.0";
    int         port       = (argc > 3) ? std::atoi(argv[3]) : 8080;

    constexpr std::size_t kArenaBudget = 64 * 1024;
    bmie::Arena arena(kArenaBudget);

    std::fprintf(stderr, "[main] arena budget: %zu bytes\n", arena.capacity());


    bmie::Model* model = bmie::load_model(arena, model_path);
    if (!model) {
        std::fprintf(stderr, "[main] failed to load model from %s\n", model_path);
        return 1;
    }

    std::fprintf(stderr, "[main] model loaded: n_features=%zu, footprint=%zu bytes\n",
                 model->n_features, bmie::model_footprint(*model));
    std::fprintf(stderr, "[main] arena used: %zu / %zu bytes (%.1f%%)\n",
                 arena.used(), arena.capacity(),
                 100.0 * arena.used() / arena.capacity());

    bmie::ServerConfig cfg{host, port};
    return bmie::run_server(*model, cfg);
}