#include "whisper.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <text_to_translate>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* input_text = argv[2];
    
    std::cout << "Loading model: " << model_path << std::endl;
    std::cout.flush();
    
    struct whisper_context_params cparams = whisper_context_default_params();
    std::cout << "About to call whisper_init_from_file_with_params..." << std::endl;
    std::cout.flush();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    std::cout << "Finished whisper_init_from_file_with_params" << std::endl;
    std::cout.flush();
    
    if (!ctx) {
        std::cout << "Failed to load model!" << std::endl;
        return 1;
    }
    
    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << "Model type: " << whisper_model_type_readable(ctx) << std::endl;
    std::cout << "Vocab size: " << whisper_model_n_vocab(ctx) << std::endl;
    
    // Test tokenization
    std::cout << "\nTesting tokenization..." << std::endl;
    std::cout << "Input text: \"" << input_text << "\"" << std::endl;
    
    int n_tokens = whisper_token_count(ctx, input_text);
    std::cout << "Number of tokens: " << n_tokens << std::endl;
    
    if (n_tokens > 0) {
        std::vector<whisper_token> tokens(n_tokens);
        
        int actual_tokens = whisper_tokenize(ctx, input_text, tokens.data(), n_tokens);
        
        if (actual_tokens > 0) {
            std::cout << "Tokenization successful!" << std::endl;
            std::cout << "Tokens: ";
            for (int i = 0; i < actual_tokens; i++) {
                std::cout << tokens[i];
                if (i < actual_tokens - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            std::cout << "Token strings: ";
            for (int i = 0; i < actual_tokens; i++) {
                const char* token_str = whisper_token_to_str(ctx, tokens[i]);
                std::cout << "\"" << (token_str ? token_str : "<null>") << "\"";
                if (i < actual_tokens - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Tokenization failed!" << std::endl;
        }
    } else {
        std::cout << "No tokens generated!" << std::endl;
    }
    
    whisper_free(ctx);
    
    return 0;
} 