# Bare-metal-inference

A from-scratch ML inference engine in **C++20** with an **arena
allocator** managing every tensor in the forward pass. No BLAS, no Eigen,
no `malloc` in the inference hot path. Trained in Python, served as a
single static binary.

> Predicts whether a shipment will be delayed, given 8 features.
> Logistic regression, gradient descent, exported to a 116-byte raw
> float32 binary the C++ engine reads directly.

---

## Quick start

### Build and run

```bash
# 1. Train the model (Python)
cd trainer
pip install -r requirements.txt
python3 generate_data.py    # synthetic shipment data, 5000 rows
python3 train.py            # logistic regression, gradient descent
python3 export_weights.py   # → ../model.bin (116 bytes)
cd ..

# 2. Build the engine (C++20)
cd engine
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
cd ..

# 3. Run
./engine/build/bare_metal_inference model.bin 0.0.0.0 8080
```

### Run the test suite

```bash
cd engine/build
ctest --output-on-failure
```

Three test binaries: `test_arena` (6 cases), `test_math` (12 cases),
`test_inference` (parity against Python trainer, 5 cases + 1000-call
stress test).

### Hit the API

```bash
# Liveness probe
curl http://localhost:8080/health
# {"status":"ok"}

# Easy shipment (reliable, short lead, weekday) → on_time
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"features":[5.0, 0.95, 500.0, 0.0, 100.0, 0.0, 0.0, 0.0]}'
# {"label":"on_time","latency_us":1,"probability":0.0295}

# Hard shipment (unreliable, far, weekend) → delayed
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"features":[25.0, 0.55, 4500.0, 3.0, 4000.0, 4.0, 2.0, 1.0]}'
# {"label":"delayed","latency_us":1,"probability":0.8080}

# In-process latency benchmark
curl 'http://localhost:8080/bench?n=10000'
# {"max_us":8,"n":10000,"p50_us":0,"p95_us":1,"p99_us":1,"sink":...}
```

---

## Architecture

```
╔════════════════════════════════════════════════════════════════════╗
║  TRAINING  (Python — runs once at build time)                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  generate_data.py              train.py           export_weights.py║
║  ┌──────────────┐              ┌──────────────┐   ┌──────────────┐ ║
║  │ 5k rows      │   CSV        │ standardize  │   │ struct.pack  │ ║
║  │ 8 features   │ ───────────> │ gradient     │   │ raw float32  │ ║
║  │ + delayed    │              │ descent      │   │ little-endian│ ║
║  │ label        │              │ logistic     │   │              │ ║
║  └──────────────┘              └──────┬───────┘   └──────┬───────┘ ║
║                                      │ JSON               │ binary ║
║                                      ▼                    ▼        ║
║                            scaler.json                model.bin    ║
║                            weights.json               (116 bytes)  ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║  SERVING   (C++ — runs forever, handles HTTP)                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║                       main.cpp  (server entry)                     ║
║                              │                                     ║
║                              ▼                                     ║
║                    ┌───────────────────┐                           ║
║                    │ Arena (64 KB)     │  ← one new[] at boot      ║
║                    │  ┌─────────────┐  │                           ║
║                    │  │ Model data  │  │  ← loaded from model.bin  ║
║                    │  └─────────────┘  │     in load_model()       ║
║                    └─────────┬─────────┘                           ║
║                              │                                     ║
║                              ▼                                     ║
║                    ┌───────────────────┐                           ║
║                    │ HTTP server       │  cpp-httplib              ║
║                    │ :8080             │  POST /predict            ║
║                    └───────────────────┘                           ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

**The bridge between Python and C++ is `model.bin`.** 116 bytes on disk.
No protocol buffers, no JSON parsing at runtime, no shared memory.
Just bytes both sides agree on.

---

## Memory layout at runtime

After `load_model` returns, the 64 KB arena holds the entire model:

```
   offset  0  ┌──────────────┐
              │  mean[8]     │   32 bytes  (8 float32)
              │              │
   offset 32  ├──────────────┤
              │  std[8]      │   32 bytes
              │              │
   offset 64  ├──────────────┤
              │  weights[8]  │   32 bytes
              │              │
   offset 96  ├──────────────┤
              │  bias        │    4 bytes
   offset 100 ├──────────────┤
              │  Model{}     │  ~40 bytes (n_features, 4 ptrs, bias)
   offset 140 ├──────────────┤
              │              │
              │  (free)      │   ~65,396 bytes available
              │              │
              ▼              ▼
   offset 65536  end of arena

   used:  140 / 65536 bytes   (0.2% of budget)
```

After startup, the arena is **frozen**. No more allocations. Every
byte the model will ever need is already there. The request handler
just reads from these addresses.

---

## The per-request path

```
client sends JSON                                server responds
{"features": [8 floats]}                         {"probability": 0.73,
                                                  "label": "delayed",
                                                  "latency_us": 1}

       │                                               ▲
       │ JSON                                          │ JSON
       ▼                                               │
┌─────────────────┐                              ┌─────┴──────┐
│ api.cpp         │                              │ api.cpp    │
│ parse JSON,     │ ───── features[8] ────────> │ build JSON │
│ extract floats  │                              │            │
└─────────────────┘                              └────────────┘
                                                        ▲
                                                        │ probability
                                                        │
                                            ┌───────────┴─────────┐
                                            │ model.cpp           │
                                            │ predict(m, x)       │
                                            │                     │
                                            │ ┌─────────────────┐ │
                                            │ │ STACK:          │ │
                                            │ │ float x[8]      │ │  ← 32 bytes,
                                            │ └─────────────────┘ │    freed on return
                                            │                     │
                                            │ normalize(x, ...)   │  reads ARENA,
                                            │                     │  writes STACK
                                            │                     │
                                            │ gemv(&z, w, x, b)   │  reads ARENA + STACK
                                            │                     │
                                            │ return sigmoid(z)   │  pure compute
                                            └─────────────────────┘
                                                        ▲
                                                        │ reads from
                                            ┌───────────┴─────────┐
                                            │     ARENA           │
                                            │ (frozen since       │
                                            │  load_model)        │
                                            └─────────────────────┘
```

**The arena holds the model. The stack holds the request. The heap holds
nothing.** That sentence is the whole project.


## Endpoints

| method | path                | body                              | response                                          |
|---|---|---|---|
| `GET`  | `/`                 | —                                 | usage info (JSON)                                 |
| `GET`  | `/health`           | —                                 | `{"status":"ok"}`                                 |
| `POST` | `/predict`          | `{"features": [8 floats]}`        | `{"probability": float, "label": "delayed"\|"on_time", "latency_us": int}` |
| `GET`  | `/bench?n=<int>`    | —                                 | `{"n": int, "p50_us": int, "p95_us": int, "p99_us": int, "max_us": int, "sink": float}` |

### Feature order

| index | name                 | type    | range             |
|---|---|---|---|
| 0 | `lead_time_days`     | float   | days, 1..30         |
| 1 | `supplier_reliability` | float | 0..1               |
| 2 | `distance_km`        | float   | km, 50..5000        |
| 3 | `season_q`           | int     | 0..3                |
| 4 | `order_value`        | float   | dollars, 10..5000   |
| 5 | `item_category`      | int     | 0..4                |
| 6 | `carrier`            | int     | 0..2                |
| 7 | `weekend_flag`       | int     | 0 or 1              |

Order must match `trainer/generate_data.py` and the JSON files. The
contract is enforced by `test_inference`.

---
## Project layout

```
bare-metal-inference/
├── trainer/
│   ├── generate_data.py     # synthetic shipment data
│   ├── train.py             # logistic regression, gradient descent
│   ├── export_weights.py    # → model.bin
│   ├── plot_loss.py         # loss curve PNG for the README
│   └── requirements.txt
├── engine/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── arena.hpp        # hand-rolled bump allocator
│   │   ├── math.hpp         # hand-rolled gemv, sigmoid, normalize
│   │   ├── model.hpp        # load model.bin, predict()
│   │   └── api.hpp          # run_server()
│   ├── src/
│   │   ├── model.cpp
│   │   ├── api.cpp
│   │   ├── httplib_impl.cpp # CPPHTTPLIB_IMPL switch
│   │   └── main.cpp
│   ├── tests/
│   │   ├── test_arena.cpp
│   │   ├── test_math.cpp
│   │   └── test_inference.cpp
│   └── third_party/         # vendored single-header libs
│       ├── httplib.h
│       └── nlohmann/json.hpp
├── model.bin                # generated by export_weights.py
└── README.md
```

---

## License

MIT