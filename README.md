<div align="center">

# LangBridge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple.svg)](https://www.anthropic.com/)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-red.svg)](https://en.cppreference.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Multi-Model AI Platform: Convert Python to High-Performance C++ using Frontier & Open-Source Models**

[Features](#features) • [Supported Models](#supported-models) • [Installation](#installation) • [Usage](#usage) • [Performance](#performance-benchmarks)

</div>

---

## 📋 Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Architecture](#architecture)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🤖 **Multi-Model Support**

#### Frontier AI Models
- **OpenAI GPT-4o-mini** - Fast code conversion with good optimization
- **Anthropic Claude Sonnet 4** - Superior optimization, better type handling
- **Google Gemini 2.5 Pro** - Advanced code understanding and generation

#### Open-Source Models (HuggingFace)
- **CodeQwen 1.5 7B** - Specialized for code understanding
- **CodeLlama 70B** - Large-scale code generation with instruction following
- **StarCoder2 15B** - Fast and efficient code generation
- **CodeGemma 7B** - Lightweight Google model for efficient inference

### ⚡ **Real-Time Code Execution**
- Execute Python code directly with captured output
- Compile and run C++ with optimized flags for Apple Silicon
- Performance timing for both languages
- Live result display in Gradio interface

### 🎯 **Advanced Optimization**
- Automatic performance optimization targeting high-performance systems
- Intelligent type conversion and overflow prevention
- Vectorization and algorithm optimization
- Compilation with aggressive optimization flags (-Ofast)
- Support for 128-bit integers for large computations

### 🖥️ **Interactive Web UI**
- Beautiful Gradio interface with syntax highlighting
- Side-by-side code comparison (Python ↔ C++)
- Color-coded output results
- Model selection dropdown for easy switching
- One-click code conversion and execution

### 📊 **Performance Analytics**
- Execution time tracking and comparison
- Comparative benchmarking between Python and C++
- Support for large-scale computations (100M+ iterations)
- Speedup metrics for code conversion

### 🔀 **Streaming Response**
- Real-time streaming for all AI models
- Progressive code generation display
- Fast feedback during code conversion

---

## 🤖 Supported Models

### Frontier AI Models (Cloud-Based)

| Model | Provider | Speed | Quality | Context | Cost |
|-------|----------|-------|---------|---------|------|
| GPT-4o-mini | OpenAI | ⚡⚡⚡ | ⭐⭐⭐⭐ | 128K | Low |
| Claude Sonnet 4 | Anthropic | ⚡⚡ | ⭐⭐⭐⭐⭐ | 200K | Medium |
| Gemini 2.5 Pro | Google | ⚡⚡ | ⭐⭐⭐⭐⭐ | 1M | Medium |

### Open-Source Models (HuggingFace Endpoints)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| CodeQwen 1.5 | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| CodeLlama | 70B | ⚡⚡ | ⭐⭐⭐⭐⭐ | High-quality code |
| StarCoder2 | 15B | ⚡⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| CodeGemma | 7B | ⚡⚡⚡ | ⭐⭐⭐ | Lightweight |

---

## 📦 Requirements

### System Requirements
- **Python 3.8+**
- **C++ 17 compiler** (clang++ recommended for Apple Silicon)
- **8GB+ RAM** (for large computations)

### Python Dependencies
```
openai>=1.0.0
anthropic>=0.7.0
google-generativeai>=0.3.0
python-dotenv>=0.21.0
gradio>=4.0.0
ipython>=8.0.0
huggingface-hub>=0.16.0
transformers>=4.30.0
```

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/langbridge.git
cd langbridge
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Frontier Models
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Open-Source Models (HuggingFace)
HF_API_KEY=your_huggingface_token_here
```

### 5. Verify C++ Compiler
```bash
clang++ --version
```

---

## ⚙️ Configuration

### API Keys Setup

#### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API keys section
3. Create new secret key
4. Add to `.env` file as `OPENAI_API_KEY`

#### Anthropic API
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Generate API key
3. Add to `.env` file as `ANTHROPIC_API_KEY`

#### Google Gemini API
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create API key
3. Add to `.env` file as `GOOGLE_API_KEY`

#### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token (read access)
3. Add to `.env` file as `HF_API_KEY`

### Model Configuration
```python
# Frontier Models
claude_model = 'claude-sonnet-4-20250514'
openai_model = 'gpt-4o-mini'
gemini_model = 'gemini-2.5-pro'

# Open-Source Models
code_qwen = "Qwen/CodeQwen1.5-7B-Chat"
code_llama = 'codellama/CodeLlama-70b-Instruct-hf'
code_star_coder = 'bigcode/starcoder2-15b'
code_gemma = 'google/codegemma-7b-it'

# HuggingFace Endpoints (optional)
CODE_QWEN_URL = 'https://your-endpoint.endpoints.huggingface.cloud'
CODE_LLAMA_URL = 'https://your-endpoint.endpoints.huggingface.cloud'
CODE_STAR_CODER_URL = 'https://your-endpoint.endpoints.huggingface.cloud'
CODE_GEMMA_URL = 'https://your-endpoint.endpoints.huggingface.cloud'
```

### Compiler Optimization Flags
```bash
clang++ -Ofast -std=c++17 -march=armv8.5-a -mtune=apple-m1 -mcpu=apple-m1
```

---

## 🎮 Usage

### Quick Start

#### 1. Launch Gradio Interface
```bash
jupyter notebook langbridge.ipynb
# Select the final UI cell and run it
```

#### 2. Using the Web UI
- **Input Python Code**: Paste your Python algorithm in the left textbox
- **Select Model**: Choose from 7 available models (GPT, Claude, Gemini, CodeQwen, CodeLlama, StarCoder, Gemma)
- **Convert**: Click "Convert code" button to generate C++ equivalent
- **Execute Python**: View Python execution results and timing
- **Execute C++**: View C++ compilation and execution results
- **Compare**: Analyze performance differences

### Model Selection Guide

**For Best Quality (Slowest):**
```
Claude Sonnet 4 > CodeLlama > Gemini 2.5 Pro > GPT-4o-mini
```

**For Fastest Speed (Decent Quality):**
```
CodeQwen > GPT-4o-mini > StarCoder > Gemma
```

**For Balanced Performance:**
```
StarCoder or GPT-4o-mini
```

### Programmatic Usage

#### Convert Code with GPT
```python
from langbridge import ui_gpt

python_code = """
def calculate(n):
    return sum(i**2 for i in range(n))
"""

for chunk in ui_gpt(python_code):
    print(chunk, end='', flush=True)
```

#### Convert Code with Claude
```python
from langbridge import ui_claude

for chunk in ui_claude(python_code):
    print(chunk, end='', flush=True)
```

#### Convert with Open-Source Models
```python
from langbridge import stream_code_qwen, stream_code_llama, stream_star_coder, stream_code_gemma

# CodeQwen
result = stream_code_qwen(python_code)
print(result)

# CodeLlama (streaming)
for chunk in stream_code_llama(python_code):
    print(chunk, end='', flush=True)

# StarCoder (streaming)
for chunk in stream_star_coder(python_code):
    print(chunk, end='', flush=True)

# CodeGemma (streaming)
for chunk in stream_code_gemma(python_code):
    print(chunk, end='', flush=True)
```

#### Execute Python Code
```python
from langbridge import execute_python

result = execute_python(python_code)
print(result)  # Captured output
```

#### Execute C++ Code
```python
from langbridge import execute_cpp

result = execute_cpp(cpp_code)
print(result)  # Compilation + execution output
```

---

## 🤖 AI Models

### System Prompt
```
You are an assistant that reimplements Python code in high performance C++ 
for an M4 Mac Pro. Respond only with C++ code; use comments sparingly and 
do not provide any explanation other than occasional comments. The C++ 
response needs to produce an identical output in the fastest possible time.
Keep implementations of random number generators identical so that results 
match perfectly.
```

### User Prompt Template
```
Rewrite this Python code in C++ with the fastest possible implementation 
that produces identical output in the least time. Respond only with C++ code; 
do not explain your work other than a few comments. Pay attention to number 
types to ensure no int overflows. Remember to #include all necessary C++ 
packages such as iomanip.
```

---

## 📊 Performance Benchmarks

### Example 1: Pi Calculation (100M iterations)

#### Python Execution
```
Result: 3.141592658589
Execution Time: 5.412465 seconds
```

#### C++ (GPT-Optimized)
```
Result: 3.141592658589
Execution Time: 0.131795 seconds
```

#### C++ (Claude-Optimized)
```
Result: 3.141592662604
Execution Time: 0.092037 seconds
```

**Performance Improvement**: **50-59x faster** than Python

### Example 2: Maximum Subarray Sum (10K array, 20 runs)

#### Python Execution
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 14.942410 seconds
```

#### C++ (GPT-Optimized)
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 0.001839 seconds
```

#### C++ (Claude-Optimized)
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 0.487203 seconds
```

**Performance Improvement**: **8100-30x faster** than Python

### Model Comparison

| Model | Conv. Speed | Code Quality | Output Match | Best Use |
|-------|------------|--------------|--------------|----------|
| GPT-4o-mini | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✓ | Fast iterations |
| Claude | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✓ | Production code |
| CodeLlama | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✓ | Complex algorithms |
| Gemini | ⚡⚡ | ⭐⭐⭐⭐ | ✓ | Edge cases |
| StarCoder | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✓ | Balanced |

---

## 🏗️ Architecture

### Component Overview

```
┌──────────────────────────────────────────────────────────┐
│            Gradio Web Interface (UI Layer)               │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌────────────────────┐      ┌────────────────────┐     │
│  │  Python Editor     │      │   C++ Editor       │     │
│  └────────────────────┘      └────────────────────┘     │
│                                                            │
│  ┌────────────────────┐      ┌────────────────────┐     │
│  │ Python Executor    │      │  C++ Executor      │     │
│  └────────────────────┘      └────────────────────┘     │
├──────────────────────────────────────────────────────────┤
│           Multi-Model Code Conversion Layer              │
├────────────────┬──────────────────┬────────────────────┤
│                │                  │                    │
│   Frontier     │  Open-Source     │  System Prompts   │
│   Models       │  Models          │                   │
│   ──────       │  ────────        │                   │
│   • OpenAI     │  • CodeQwen      │  • User Prompt    │
│   • Anthropic  │  • CodeLlama     │  • System Msg     │
│   • Google     │  • StarCoder     │                   │
│                │  • CodeGemma     │                   │
│                │                  │                   │
├────────────────┴──────────────────┴────────────────────┤
│         Execution & Compilation Layer                   │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌────────────────────┐      ┌────────────────────┐     │
│  │  Python Exec       │      │   C++ Compiler     │     │
│  │  (via exec())      │      │   (clang++)        │     │
│  └────────────────────┘      └────────────────────┘     │
│                                                            │
│  Output Capture (io.StringIO) & File Writing           │
├──────────────────────────────────────────────────────────┤
```

### Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `optimize_final()` | Route to appropriate AI model | Python code, model name | Generator yielding C++ code |
| `ui_gpt()` | Stream OpenAI GPT response | Python code | Streaming C++ chunks |
| `ui_claude()` | Stream Anthropic Claude response | Python code | Streaming C++ chunks |
| `ui_gemini()` | Stream Google Gemini response | Python code | Streaming C++ chunks |
| `stream_code_qwen()` | CodeQwen conversion | Python code | C++ code |
| `stream_code_llama()` | CodeLlama streaming conversion | Python code | Streaming C++ chunks |
| `stream_star_coder()` | StarCoder streaming conversion | Python code | Streaming C++ chunks |
| `stream_code_gemma()` | CodeGemma streaming conversion | Python code | Streaming C++ chunks |
| `execute_python()` | Run Python with output capture | Python code | Captured stdout |
| `execute_cpp()` | Compile and run C++ | C++ code | Executable output |
| `user_prompt_for()` | Build AI prompt | Python code | Formatted prompt string |
| `write_output()` | Save C++ to file | C++ code | File write confirmation |

---

## 💡 Examples

### Example 1: Simple Pi Calculation

#### Python Input
```python
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
```

#### C++ Output (Claude-Generated)
```cpp
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, int param1, int param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        long long j1 = i * param1 - param2;
        long long j2 = i * param1 + param2;
        result += (1.0 / j2) - (1.0 / j1);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100000000LL, 4, 1) * 4.0;
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double execution_time = duration.count() / 1000000.0;
    
    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << std::fixed << std::setprecision(6) << "Execution Time: " << execution_time << " seconds" << std::endl;
    
    return 0;
}
```

### Example 2: Complex Algorithm with LCG

#### Python Input (Maximum Subarray Sum)
```python
def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value

def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum
```

#### C++ Output (GPT-Optimized)
```cpp
#include <iostream>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <algorithm>

static inline uint32_t lcg_next(uint32_t& v) {
    v = v * 1664525u + 1013904223u;
    return v;
}

static __int128 max_subarray_sum(std::size_t n, uint32_t seed, long long min_val, long long max_val) {
    uint32_t state = seed;
    unsigned __int128 urange = (unsigned __int128)((__int128)max_val - (__int128)min_val + 1);
    __int128 best = 0, current = 0;
    bool first = true;
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t rnd = lcg_next(state);
        unsigned __int128 rem = (unsigned __int128)rnd % urange;
        __int128 val = (__int128)rem + (__int128)min_val;
        if (first) {
            current = best = val;
            first = false;
        } else {
            __int128 sum = current + val;
            current = (sum > val) ? sum : val;
            if (current > best) best = current;
        }
    }
    return best;
}
```

---

## 🐛 Troubleshooting

### Issue: "API Key not found"
**Solution**: Verify `.env` file exists and contains correct keys
```bash
cat .env  # Check file contents
```

### Issue: "clang++ not found"
**Solution**: Install Xcode Command Line Tools
```bash
xcode-select --install
```

### Issue: "HuggingFace authentication failed"
**Solution**: Verify HF_API_KEY in .env and ensure token has read access
```bash
huggingface-cli login  # Interactive login
```

### Issue: C++ compilation fails
**Solution**: Check for syntax errors in generated code
```bash
clang++ -std=c++17 yourfile.cpp -o out -v  # Verbose compilation
```

### Issue: Python exec() returns None in Gradio
**Solution**: Use `io.StringIO` to capture stdout (already implemented in `execute_python()`)

### Issue: Memory issues with large arrays
**Solution**: Reduce array size or increase system RAM
```bash
top -l 1 | grep Memory  # Monitor memory usage
```

### Issue: Open-Source models timing out
**Solution**: Check HuggingFace endpoint status and increase timeout
```python
# Increase max_new_tokens or reduce input size
max_new_tokens = 1000  # Adjust based on model capacity
```

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Guidelines
- Follow PEP 8 for Python code
- Add docstrings to new functions
- Update README with new features
- Test thoroughly before submitting
- Include performance benchmarks for new optimizations

---

## 📁 File Structure

```
langbridge/
├── langbridge.ipynb                    # Original single-model version
├── langbridge(Open Source + Frontier).ipynb  # Multi-model version
├── .env                                # Environment variables (git-ignored)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── optimized.cpp                       # Generated C++ files
├── optimized_practice04                # Generated C++ binary
└── LICENSE                             # MIT License
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Citation

If you use LangBridge in research, please cite:
```bibtex
@software{langbridge2025,
  author = {Asutosha Nanda},
  title = {LangBridge},
  year = {2025},
  url = {https://github.com/yourusername/langbridge}
}
```

---

## 📚 Additional Resources

- [OpenAI API Docs](https://platform.openai.com/docs/)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Google AI Docs](https://ai.google.dev/docs)
- [HuggingFace Hub](https://huggingface.co/docs)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [C++ Reference](https://en.cppreference.com/)

---

## 🙋 Support & Contact

- **Issues**: Open GitHub issues for bugs
- **Questions**: Create discussions in GitHub

---

<div align="center">

**[⬆ Back to Top](#-langbridge)**

**Multi-Model AI Code Optimizer**  
Made with ❤️ for high-performance computing

</div>
