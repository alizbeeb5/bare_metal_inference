#pragma once
#include "model.hpp"
#include <string>

namespace bmie{
    struct ServerConfig {
        std::string host ="0.0.0.0";
        int port = 8080;
    };
    int run_server(const Model& model, const ServerConfig& cfg) noexcept;
}