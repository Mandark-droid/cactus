#include "ffi/cactus_ffi.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>

struct TokenTiming {
    uint32_t token_id;
    std::string text;
    double elapsed_ms;
};

struct StreamState {
    std::vector<TokenTiming> tokens;
    std::chrono::steady_clock::time_point start;
    std::string full_text;
};

static void token_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* state = static_cast<StreamState*>(user_data);
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(now - state->start).count();

    state->tokens.push_back({token_id, token ? token : "", elapsed});
    state->full_text += (token ? token : "");

    std::cout << (token ? token : "") << std::flush;
}

static bool run_completion(cactus_model_t model, const char* label,
                           const char* messages, const char* options,
                           int expected_min_tokens = 5) {
    std::cout << "\n--- " << label << " ---\n";

    StreamState state;
    state.start = std::chrono::steady_clock::now();

    char response[8192] = {};
    int rc = cactus_complete(model, messages, response, sizeof(response),
                             options, nullptr, token_callback, &state);

    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - state.start).count();

    std::cout << "\n";

    if (rc < 0) {
        std::cerr << "[FAIL] " << label << ": cactus_complete returned " << rc << "\n";
        return false;
    }

    size_t n = state.tokens.size();
    if (n < static_cast<size_t>(expected_min_tokens)) {
        std::cerr << "[FAIL] " << label << ": only " << n << " tokens (expected >= " << expected_min_tokens << ")\n";
        return false;
    }

    double ttft = state.tokens.empty() ? 0.0 : state.tokens[0].elapsed_ms;
    double decode_ms = n > 1 ? (state.tokens.back().elapsed_ms - ttft) / (n - 1) : 0.0;
    double tps = decode_ms > 0 ? 1000.0 / decode_ms : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[METRICS] tokens=" << n
              << "  ttft=" << ttft << "ms"
              << "  decode=" << decode_ms << "ms/tok"
              << "  tps=" << tps
              << "  total=" << total_ms << "ms\n";

    std::cout << "[PASS] " << label << "\n";
    return true;
}

int main(int argc, char** argv) {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path && argc > 1) model_path = argv[1];
    if (!model_path) {
        std::cerr << "Usage: test_qwen3_moe <model_path>\n"
                  << "  or set CACTUS_TEST_MODEL env var\n";
        return 1;
    }

    std::cout << "=== Qwen3 MoE Phase 3 Validation ===\n";
    std::cout << "Model: " << model_path << "\n\n";

    std::cout << "[INIT] Loading model...\n";
    auto t0 = std::chrono::steady_clock::now();
    cactus_model_t model = cactus_init(model_path, nullptr, false);
    auto t1 = std::chrono::steady_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!model) {
        std::cerr << "[FAIL] Model init failed\n";
        return 1;
    }
    std::cout << "[INIT] OK (" << std::fixed << std::setprecision(0) << init_ms << "ms)\n";

    int pass = 0, fail = 0;

    // T1: Greedy single-token (temperature=0)
    {
        const char* messages = R"([
            {"role": "user", "content": "Hi"}
        ])";
        const char* options = R"({
            "max_tokens": 32,
            "temperature": 0,
            "confidence_threshold": 0,
            "stop_sequences": ["<|im_end|>"]
        })";
        if (run_completion(model, "T1: Greedy (temp=0)", messages, options, 2))
            pass++;
        else
            fail++;
        cactus_reset(model);
    }

    // T2: Chat with system prompt
    {
        const char* messages = R"([
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"}
        ])";
        const char* options = R"({
            "max_tokens": 64,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "confidence_threshold": 0,
            "stop_sequences": ["<|im_end|>"]
        })";
        if (run_completion(model, "T2: Chat (2+2)", messages, options, 3))
            pass++;
        else
            fail++;
        cactus_reset(model);
    }

    // T3: Longer generation
    {
        const char* messages = R"([
            {"role": "user", "content": "Explain what a neural network is in simple terms."}
        ])";
        const char* options = R"({
            "max_tokens": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "confidence_threshold": 0,
            "stop_sequences": ["<|im_end|>"]
        })";
        if (run_completion(model, "T3: Longer generation (128 tokens)", messages, options, 20))
            pass++;
        else
            fail++;
        cactus_reset(model);
    }

    // T4: Multi-turn conversation
    {
        const char* messages = R"([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
            {"role": "user", "content": "What is my name?"}
        ])";
        const char* options = R"({
            "max_tokens": 32,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "confidence_threshold": 0,
            "stop_sequences": ["<|im_end|>"]
        })";
        if (run_completion(model, "T4: Multi-turn (name recall)", messages, options, 2))
            pass++;
        else
            fail++;
        cactus_reset(model);
    }

    // T5: Decode stability (50 tokens)
    {
        const char* messages = R"([
            {"role": "user", "content": "Write a short story about a robot."}
        ])";
        const char* options = R"({
            "max_tokens": 50,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "confidence_threshold": 0,
            "stop_sequences": ["<|im_end|>"]
        })";
        if (run_completion(model, "T5: Decode stability (50 tokens)", messages, options, 15))
            pass++;
        else
            fail++;
        cactus_reset(model);
    }

    cactus_destroy(model);

    std::cout << "\n=== RESULTS: " << pass << "/" << (pass + fail) << " passed ===\n";
    return fail > 0 ? 1 : 0;
}
