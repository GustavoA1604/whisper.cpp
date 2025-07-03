# Current test status

The following instructions are for linux and might need to be adapted for other platforms

## Conversion script

Install dependencies
```
pip install numpy torch transformers sentencepiece
```

Then, assuming that you have an opus-mt-* repository, you can convert it to .ggml format using:
```
python convert_opus_to_ggml.py --model-dir opus-mt-en-it --output ggml-opus-en-it.bin
```

## Building and testing with Marian changes

First install new dependencies
```
sudo apt-get update
sudo apt-get install -y cmake build-essential pkg-config libprotobuf-dev protobuf-compiler

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../..
```

Then build the main whisper.cpp application:
```
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DWHISPER_ALL_WARNINGS=ON
cmake --build build-debug -j
```

Finally, build marian_test.cpp:
```
g++ -I./include -I./ggml/include -L./build-debug/src -L./build-debug/ggml/src -o test_marian test_marian.cpp -lwhisper -lggml -lggml-base -lggml-cpu -pthread -std=c++17
```

Test it with the following command, which at this point only runs the tokenization code
```
LD_LIBRARY_PATH=./build-debug/src:./build-debug/ggml/src ./test_marian ggml-opus-en-it.bin "Hello world"
```

It should have an output similar to the following
```
...
Model type: marian
Vocab size: 80035

Testing tokenization...
Input text: "Hello world"
whisper_tokenize: too many resulting tokens: 3 (max 0)
Number of tokens: 3
Tokenization successful!
Tokens: 226, 1127, 499
Token strings: "▁H", "ello", "▁world"
```


