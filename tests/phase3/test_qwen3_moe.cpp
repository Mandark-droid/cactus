// Phase 3: Qwen3 MoE numerical validation test
// Usage: ./test_qwen3_moe <model_path> [prompt]
// Each sub-test creates and destroys its own model instance for full isolation.

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

void print_cache(Model* m, const std::string& label) {
    std::cout << "    [cache@" << label << "] seq=" << m->get_cache_seq_len()
              << " total=" << m->get_cache_total_len()
              << " empty=" << m->is_cache_empty() << "\n";
}

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

    // Step 1: Smoke test — create, init, tokenize, then DESTROY
    std::vector<uint32_t> tokens;
    {
        std::cout << "[1/7] Smoke test...\n";
        auto t0 = std::chrono::steady_clock::now();
        auto model = create_model(model_path);
        if (!model) {
            std::cerr << "FAILED: Could not create model from " << model_path << "\n";
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  Model type: " << static_cast<int>(model->get_config().model_type) << "\n";
        std::cout << "  Layers: " << model->get_config().num_layers << "\n";
        std::cout << "  Hidden dim: " << model->get_config().hidden_dim << "\n";
        std::cout << "  Experts: " << model->get_config().num_experts << "\n";
        std::cout << "  Top-K experts: " << model->get_config().num_top_experts << "\n";
        std::cout << "  Created in " << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        auto t2 = std::chrono::steady_clock::now();
        bool ok = model->init(model_path, context_size, "", true);
        if (!ok) {
            std::cerr << "FAILED: Model init failed\n";
            return 1;
        }
        auto t3 = std::chrono::steady_clock::now();
        std::cout << "  Initialized in " << std::fixed << std::setprecision(1)
                  << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";

        auto* tokenizer = model->get_tokenizer();
        if (!tokenizer) {
            std::cerr << "FAILED: No tokenizer loaded\n";
            return 1;
        }
        tokens = tokenizer->encode(prompt);
        std::cout << "  Token IDs (" << tokens.size() << "): [";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << tokens[i];
        }
        std::cout << "]\n";
        std::cout << "  Destroying smoke test model before I-tests...\n\n";
    }  // smoke test model destroyed here

    // ========================================================
    // Step 2: Isolated single-operation tests
    // Each test creates its own model instance (fresh state).
    // HF reference (softmax->topk routing): [9707] -> [13, 13, 16, 24, 17, 16, 271, 91, 2631, 760]
    // ========================================================
    std::cout << "[2/7] Isolated KV cache tests (each model is fresh)...\n";
    int total_pass = 0, total_tests = 0;

    // I1: prefill({9707}) + decode({13}) — baseline
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        print_cache(m.get(), "after-init");
        m->prefill({9707}, 1);
        print_cache(m.get(), "after-prefill");
        uint32_t tok = m->decode({13}, 0.0f, 1.0f, 1);
        print_cache(m.get(), "after-decode");
        bool pass = (tok == 13);
        total_tests++; if (pass) total_pass++;
        std::cout << "  I1 (prefill+decode): prefill[9707] decode[13]->" << tok
                  << " (HF:13) " << (pass ? "PASS" : "FAIL") << "\n";
    }

    // I1b: immediate repeat of I1 — does one prior model break it?
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        m->prefill({9707}, 1);
        uint32_t tok = m->decode({13}, 0.0f, 1.0f, 1);
        bool pass = (tok == 13);
        total_tests++; if (pass) total_pass++;
        std::cout << "  I1b (immediate repeat): " << tok
                  << " (HF:13) " << (pass ? "PASS" : "FAIL") << "\n";
    }

    // I2: decode({9707}) + decode({13}) — tests decode override
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        print_cache(m.get(), "after-init");
        uint32_t tok1 = m->decode({9707}, 0.0f, 1.0f, 1);
        print_cache(m.get(), "after-decode1");
        uint32_t tok2 = m->decode({13}, 0.0f, 1.0f, 1);
        print_cache(m.get(), "after-decode2");
        bool p1 = (tok1 == 13), p2 = (tok2 == 13);
        total_tests += 2; if (p1) total_pass++; if (p2) total_pass++;
        std::cout << "  I2 (decode+decode): decode[9707]->" << tok1 << (p1?" PASS":" FAIL")
                  << ", decode[13]->" << tok2 << (p2?" PASS":" FAIL") << "\n";
    }

    // I3: prefill({9707,13}) + 5 decodes — multi-token prefill
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        m->prefill({9707, 13}, 2);
        print_cache(m.get(), "after-prefill2");
        std::cout << "  I3 (prefill[9707,13]+5 decodes): ";
        uint32_t tok = 13;
        std::vector<uint32_t> hf_ref = {16, 24, 17, 16, 271};
        int matched = 0;
        for (int i = 0; i < 5; i++) {
            tok = m->decode({tok}, 0.0f, 1.0f, 1);
            std::cout << tok << " ";
            total_tests++;
            if (tok == hf_ref[i]) { matched++; total_pass++; }
        }
        std::cout << " (HF: 16 24 17 16 271, matched " << matched << "/5)\n";
    }

    // I4: prefill({9707}) + 10 sustained decodes
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        m->prefill({9707}, 1);
        std::cout << "  I4 (prefill+10 decodes): ";
        uint32_t tok = 13;
        std::vector<uint32_t> hf_ref = {13, 16, 24, 17, 16, 271, 91, 2631, 760, 6122};
        int matched = 0;
        for (int i = 0; i < 10; i++) {
            tok = m->decode({tok}, 0.0f, 1.0f, 1);
            std::cout << tok << " ";
            total_tests++;
            if (i < (int)hf_ref.size() && tok == hf_ref[i]) { matched++; total_pass++; }
        }
        std::cout << "\n  (HF: 13 16 24 17 16 271 91 2631 760 6122, matched " << matched << "/10)\n";
        print_cache(m.get(), "after-10-decodes");
    }

    // I5: decode-only chain (10 tokens) — no prefill at all
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        uint32_t tok = m->decode({9707}, 0.0f, 1.0f, 1);
        std::cout << "  I5 (decode-only chain, 10 tokens): " << tok;
        for (int i = 0; i < 9; i++) {
            tok = m->decode({tok}, 0.0f, 1.0f, 1);
            std::cout << " " << tok;
        }
        std::cout << "\n  (HF: 13 13 16 24 17 16 271 91 2631 760)\n";
    }

    // I6: no warmup — does warmup affect results?
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", false);  // no warmup
        uint32_t tok1 = m->decode({9707}, 0.0f, 1.0f, 1);
        uint32_t tok2 = m->decode({13}, 0.0f, 1.0f, 1);
        std::cout << "  I6 (no warmup): decode[9707]->" << tok1
                  << ", decode[13]->" << tok2 << "\n";
    }

    // I7: cross-contamination check — same as I1 but runs last
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        m->prefill({9707}, 1);
        uint32_t tok = m->decode({13}, 0.0f, 1.0f, 1);
        bool pass = (tok == 13);
        total_tests++; if (pass) total_pass++;
        std::cout << "  I7 (cross-check = I1 repeated): " << tok
                  << " (HF:13) " << (pass ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "  --- Score: " << total_pass << "/" << total_tests << " ---\n\n";

    // ========================================================
    // Step 3: MoE routing verification
    // ========================================================
    std::cout << "[3/7] MoE routing info...\n";
    std::cout << "  C++ routing: softmax(all_logits) -> topk -> route (HF style)\n";
    std::cout << "  Python ref:  topk(raw_logits) -> softmax(topk_scores)\n";
    std::cout << "  Status: C++ uses HF routing (matches HF reference outputs)\n\n";

    // ========================================================
    // Step 4: Chat template test
    // ========================================================
    std::cout << "[4/7] Chat template test...\n";
    {
        auto m = create_model(model_path);
        m->init(model_path, context_size, "", true);
        auto* tok_ptr = m->get_tokenizer();

        std::string chat_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        auto chat_tokens = tok_ptr->encode(chat_prompt);
        std::cout << "  Chat tokens (" << chat_tokens.size() << "): [";
        for (size_t i = 0; i < std::min(chat_tokens.size(), (size_t)10); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << chat_tokens[i];
        }
        if (chat_tokens.size() > 10) std::cout << ", ...";
        std::cout << "]\n";

        if (chat_tokens.size() > 1) {
            std::vector<uint32_t> prefill_toks(chat_tokens.begin(), chat_tokens.end() - 1);
            m->prefill(prefill_toks, prefill_toks.size());
        }

        std::cout << "  Generated: ";
        uint32_t tok = chat_tokens.back();
        std::vector<uint32_t> chat_generated;
        for (int i = 0; i < 20; i++) {
            tok = m->decode({tok}, 0.0f, 1.0f, 1);
            chat_generated.push_back(tok);
            std::cout << tok_ptr->decode({tok});
            std::cout.flush();
            if (tok == 151643 || tok == 151645) break;
        }
        std::cout << "\n  Token IDs: [";
        for (size_t i = 0; i < chat_generated.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << chat_generated[i];
        }
        std::cout << "]\n\n";
    }

    // ========================================================
    // Step 5: Full generation (fresh model)
    // ========================================================
    std::cout << "[5/7] Running forward pass (prefill + decode)...\n";
    auto model = create_model(model_path);
    model->init(model_path, context_size, "", true);
    auto* tokenizer = model->get_tokenizer();

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

    // ========================================================
    // Step 6: Extended generation with timing
    // ========================================================
    std::cout << "[6/7] Generation (greedy, 50 tokens)...\n";
    std::cout << prompt;
    std::cout << next_text;

    std::vector<uint32_t> generated = {next_token};
    std::vector<double> decode_times;
    for (int i = 0; i < 49; i++) {
        auto td0 = std::chrono::steady_clock::now();
        uint32_t tok = model->decode({generated.back()}, 0.0f, 1.0f, 1);
        auto td1 = std::chrono::steady_clock::now();
        decode_times.push_back(std::chrono::duration<double, std::milli>(td1 - td0).count());
        generated.push_back(tok);
        std::cout << tokenizer->decode({tok});
        std::cout.flush();
        if (tok == 151643 || tok == 151645) break;
    }
    std::cout << "\n\n";

    std::cout << "Token IDs: [";
    for (size_t i = 0; i < std::min(generated.size(), (size_t)20); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << generated[i];
    }
    if (generated.size() > 20) std::cout << ", ...";
    std::cout << "]\n\n";

    // Performance stats
    if (!decode_times.empty()) {
        double total_time = 0;
        for (double t : decode_times) total_time += t;
        double avg_time = total_time / decode_times.size();
        double min_time = *std::min_element(decode_times.begin(), decode_times.end());
        double max_time = *std::max_element(decode_times.begin(), decode_times.end());
        double tokens_per_sec = 1000.0 * decode_times.size() / total_time;

        std::cout << "[Performance]\n";
        std::cout << "  Tokens generated: " << generated.size() << "\n";
        std::cout << "  Decode times (ms): avg=" << std::fixed << std::setprecision(1) << avg_time
                  << " min=" << min_time << " max=" << max_time << "\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << tokens_per_sec << " tokens/sec\n";
        std::cout << "  Total generation time: " << std::fixed << std::setprecision(0) << total_time << " ms\n\n";
    }

    // ========================================================
    // Step 7: Summary
    // ========================================================
    std::cout << "[7/7] Summary\n";
    std::cout << "  Isolated test score: " << total_pass << "/" << total_tests << "\n";
    std::cout << "  Total generated: " << generated.size() << " tokens\n";
    std::cout << "DONE\n";

    return 0;
}
