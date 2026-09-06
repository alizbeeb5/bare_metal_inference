#include "api.hpp"
#include "math.hpp"
#include "model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "httplib.h"
#include <nlohmann/json.hpp>

namespace {
    using json = nlohmann::json;


    int64_t now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    constexpr const char* kUsage = R"({
      "name": "bare-metal-inference",
      "endpoints": {
        "GET  /health":  "liveness probe",
        "POST /predict": "{\"features\": [8 floats]} -> {probability, label, latency_us}",
        "GET  /bench?n=1000": "in-process latency benchmark"
      },
      "n_features": 8
    })";

    bool extract_features(const json& body, const bmie::Model& model ,std::vector<float>& out) {
        if (!body.contains("features")||!body["features"].is_array()) {
            return false;
        }
        const auto& arr = body["features"];
        if (arr.size()!=model.n_features) {
            return false;
        }
        out.clear();
        out.reserve(model.n_features);
        for (const  auto&v : arr) {
            if (!v.is_number()) return false;
            out.push_back(v.get<float>());
        }
        return true;
    }

    void handle_predict(const httplib::Request& req, httplib::Response& res, const bmie::Model& model) {
        json body ;
        try {
            body = json::parse(req.body) ;
        }
        catch (const std::exception& e) {
            res.status = 400 ;
            res.set_content(
                json{{"error", std::string{"invalid json"}+e.what()}}.dump(),"application/json"
                );
            return ;
        }

        std::vector<float> features;
        if (!extract_features(body, model, features)) {
            res.status = 400 ;
            res.set_content(
                json{{"error", "expected{\"features\": [<n_features> floats]}"}}.dump(),"application/json"
                );
            return ;
        }
        const int64_t t0 = now_us();
        const float p = predict(model, features.data());
        const int64_t t1 = now_us();

        const char* label = (p>=0.5f) ? "delayed" : "on_time";
        json out={
            {"probability",p},
            {"label",label},
            {"latency_us",t1-t0},
        };

        res.set_content(out.dump(),"application/json");
    }

    void handle_health(const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})","application/json");
    }

    void handle_bench(const httplib::Request& req, httplib::Response& res,
                  const bmie::Model& model) {

        int n = 1000;
        if (req.has_param("n")) {
            try { n = std::max(1, std::min(1'000'000,
                                           std::stoi(req.get_param_value("n")))); }
            catch (...) { n = 1000; }
        }


        std::vector<float> features(model.n_features, 0.5f);
        std::vector<int64_t> samples;
        samples.reserve(n);


        float sink = 0.0f;
        for (int i = 0; i < n; ++i) {
            const int64_t t0 = now_us();
            sink += predict(model, features.data());
            const int64_t t1 = now_us();
            samples.push_back(t1 - t0);
        }


        std::sort(samples.begin(), samples.end());
        auto pct = [&](double p) {
            const std::size_t idx = static_cast<std::size_t>(p * (samples.size() - 1));
            return samples[idx];
        };

        json out = {
            {"n",      n},
            {"p50_us", pct(0.50)},
            {"p95_us", pct(0.95)},
            {"p99_us", pct(0.99)},
            {"max_us", samples.back()},
            {"sink",   sink},  // dead-code defense
        };
        res.set_content(out.dump(), "application/json");
    }



}
namespace bmie {
    int run_server(const bmie::Model& model, const bmie::ServerConfig& cfg) noexcept {
        httplib::Server svr;

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(kUsage, "application/json");
        });
        svr.Get("/health", handle_health);
        svr.Post("/predict", [&](const httplib::Request& req, httplib::Response& res) {
            handle_predict(req, res, model);
        });
        svr.Get("/bench", [&](const httplib::Request& req, httplib::Response& res) {
            handle_bench(req, res, model);
        });

        std::fprintf(stderr, "[api] listening on %s:%d (n_features=%zu)\n",
                     cfg.host.c_str(), cfg.port, model.n_features);

        if (!svr.listen(cfg.host.c_str(), cfg.port)) {
            std::fprintf(stderr, "[api] listen failed (port in use?)\n");
            return 1;
        }
        return 0;
    }
}