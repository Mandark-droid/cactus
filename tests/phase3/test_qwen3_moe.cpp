// Phase 3: Qwen3 MoE numerical validation test
// Usage: ./test_qwen3_moe <model_path> [prompt]
// Loads the converted model, runs a forward pass, and prints logits for comparison
// with HuggingFace reference outputs.

#include "engine/engine.h"
#include "graph/graph.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace cactus::engine;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]\n";
        std::cerr << "Example: " << argv[0] << " /data/local/tmp/loggenix-moe \"Hello\"\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    const size_t context_size = 512;

    std::cout << "=== Qwen3 MoE Phase 3 Validation ===\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Prompt: \"" << prompt << "\"\n\n";

    // Step 1: Create model from config
    std::cout << "[1/5] Creating model...\n";
    auto t0 = std::chrono::steady_clock::now();
    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "FAILED: Could not create model from " << model_path << "\n";
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Model type: " << static_cast<int>(model->get_config().model_type) << "\n";
    std::cout << "  Layers: " << model->get_config().num_layers << "\n";
    std::cout << "  Hidden dim: " << model->get_config().hidden_dim << "\n";
    std::cout << "  Experts: " << model->get_config().num_experts << "\n";
    std::cout << "  Top-K experts: " << model->get_config().num_top_experts << "\n";
    std::cout << "  Created in " << std::fixed << std::setprecision(1) << create_ms << " ms\n\n";

    // Step 2: Initialize (load weights, build graph, warmup)
    std::cout << "[2/5] Initializing (loading weights, building graph)...\n";
    auto t2 = std::chrono::steady_clock::now();
    bool ok = model->init(model_path, context_size, "", true);
    if (!ok) {
        std::cerr << "FAILED: Model init failed\n";
        return 1;
    }
    auto t3 = std::chrono::steady_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  Initialized in " << std::fixed << std::setprecision(1) << init_ms << " ms\n\n";

    // Step 3: Tokenize prompt
    std::cout << "[3/5] Tokenizing prompt...\n";
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "FAILED: No tokenizer loaded\n";
        return 1;
    }
    auto tokens = tokenizer->encode(prompt);
    std::cout << "  Token IDs (" << tokens.size() << "): [";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]\n\n";

    // Step 4: KV cache path comparison
    std::cout << "[4/5] KV cache A/B/C test...\n";
    {
        // HF reference: [9707] -> [13, 13, 16, 24, 17, 16, 271, 91, ...]

        // Test A: prefill({9707}) then decode({13}) — KV from prefill
        auto modelA = create_model(model_path);
        modelA->init(model_path, context_size, "", true);
        modelA->prefill({9707}, 1);
        uint32_t tokA = modelA->decode({13}, 0.0f, 1.0f, 1);
        std::cout << "  A (prefill+decode): prefill[9707], decode[13] -> " << tokA
                  << " \"" << tokenizer->decode({tokA}) << "\"\n";

        // Test B: decode({9707}) then decode({13}) — KV from decode
        auto modelB = create_model(model_path);
        modelB->init(model_path, context_size, "", true);
        uint32_t tokB1 = modelB->decode({9707}, 0.0f, 1.0f, 1);
        uint32_t tokB2 = modelB->decode({13}, 0.0f, 1.0f, 1);
        std::cout << "  B (decode+decode): decode[9707]->" << tokB1
                  << ", decode[13]->" << tokB2 << "\n";

        // Test C: prefill({9707,13}) then 5 decodes — all KV from prefill
        auto modelC = create_model(model_path);
        modelC->init(model_path, context_size, "", true);
        modelC->prefill({9707, 13}, 2);
        std::cout << "  C (prefill[9707,13]+5 decodes): ";
        uint32_t tok = 13;  // HF step 1 token
        for (int i = 0; i < 5; i++) {
            tok = modelC->decode({tok}, 0.0f, 1.0f, 1);
            std::cout << tok << " ";
        }
        std::cout << "\n";
        std::cout << "  (HF: 16 24 17 16 271)\n\n";
    }

    // Step 5: Full generation with cache
    std::cout << "[5/5] Running forward pass (prefill + decode)...\n";
    auto t4 = std::chrono::steady_clock::now();

    if (tokens.size() > 1) {
        std::vector<uint32_t> prefill_tokens(tokens.begin(), tokens.end() - 1);
        model->prefill(prefill_tokens, prefill_tokens.size());
    }

    std::vector<uint32_t> decode_input = {tokens.back()};
    float entropy = 0.0f;
    uint32_t next_token = model->decode(decode_input, 0.0f, 1.0f, 1, "", &entropy);

    auto t5 = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    std::string next_text = tokenizer->decode({next_token});
    std::cout << "  Next token ID: " << next_token << "\n";
    std::cout << "  Next token text: \"" << next_text << "\"\n";
    std::cout << "  Entropy: " << std::fixed << std::setprecision(4) << entropy << "\n";
    std::cout << "  Forward pass: " << std::fixed << std::setprecision(1) << decode_ms << " ms\n\n";

    // Generate more tokens
    std::cout << "=== Generation (greedy, 20 tokens) ===\n";
    std::cout << prompt;
    std::cout << next_text;

    std::vector<uint32_t> generated = {next_token};
    for (int i = 0; i < 19; i++) {
        uint32_t tok = model->decode({generated.back()}, 0.0f, 1.0f, 1);
        generated.push_back(tok);
        std::cout << tokenizer->decode({tok});
        std::cout.flush();
        if (tok == 151643 || tok == 151645) break;
    }
    std::cout << "\n\n";

    std::cout << "=== Generated Token IDs ===\n[";
    for (size_t i = 0; i < generated.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << generated[i];
    }
    std::cout << "]\n\n";

    std::cout << "=== Summary ===\n";
    std::cout << "Model create: " << std::fixed << std::setprecision(1) << create_ms << " ms\n";
    std::cout << "Model init:   " << init_ms << " ms\n";
    std::cout << "First decode: " << decode_ms << " ms\n";
    std::cout << "Total tokens: " << generated.size() << "\n";
    std::cout << "DONE\n";

    return 0;
}
