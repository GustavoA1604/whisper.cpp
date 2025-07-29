#include "whisper.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <text_to_translate>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* input_text = argv[2];
    
    std::cout << "Loading IndicTrans2 model: " << model_path << std::endl;
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
    
    // Verify this is an IndicTrans2 model
    if (strcmp(whisper_model_type_readable(ctx), "indictrans") != 0) {
        std::cout << "Warning: Expected IndicTrans2 model, but got: " << whisper_model_type_readable(ctx) << std::endl;
    }
    
    // Run full IndicTrans2 translation pipeline
    std::cout << "\nRunning IndicTrans2 translation..." << std::endl;
    std::cout << "Input text: \"" << input_text << "\"" << std::endl;
    
    int result = indictrans_full(ctx, input_text);
    
    if (result == 0) {
        std::cout << "\nTranslation completed successfully!" << std::endl;
        
        // Get the number of segments (results)
        int n_segments = whisper_full_n_segments(ctx);
        std::cout << "Number of segments: " << n_segments << std::endl;
        
        // Print each segment
        for (int i = 0; i < n_segments; i++) {
            const char* segment_text = whisper_full_get_segment_text(ctx, i);
            std::cout << "Segment " << i << ": \"" << (segment_text ? segment_text : "<null>") << "\"" << std::endl;
        }
    } else if (result == -1) {
        std::cout << "Translation failed: Encoder error" << std::endl;
    } else if (result == -2) {
        std::cout << "Translation failed: Decoder error" << std::endl;
    } else {
        std::cout << "Translation failed with unknown error code: " << result << std::endl;
    }
    
    whisper_free(ctx);
    
    return 0;
} 