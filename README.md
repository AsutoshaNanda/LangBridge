<div align="center">

# LangBridge 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://platform.openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple.svg)](https://www.anthropic.com/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-red.svg)](https://en.cppreference.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Intelligent Python to C++ code conversion using AI with real-time execution and performance comparison**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [API Models](#api-models) • [Performance](#performance)

</div>

---
<h2 align="center">📋 Table of Contents</h2>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-models">API Models</a> •
  <a href="#performance-benchmarks">Performance Benchmarks</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#examples">Examples</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>


---

## ✨ Features

### 🔄 **Multi-Model Support**
- **OpenAI GPT-4o-mini** for rapid code conversion
- **Anthropic Claude Sonnet 4** for optimized implementations
- Model selection via interactive UI dropdown

### ⚡ **Real-Time Code Execution**
- Execute Python code directly with captured output
- Compile and run C++ with optimized flags for Apple M1/M4 Mac Pro
- Performance timing for both languages
- Live result display in Gradio interface

### 🎯 **Optimization Focus**
- Automatic performance optimization targeting high-performance systems
- Intelligent type conversion and overflow prevention
- Vectorization and algorithm optimization
- Compilation with aggressive optimization flags (-Ofast)

### 🖥️ **Interactive Web UI**
- Beautiful Gradio interface with syntax highlighting
- Side-by-side code comparison (Python ↔ C++)
- Color-coded output results
- One-click code conversion and execution

### 📊 **Performance Analytics**
- Execution time tracking
- Comparative benchmarking between Python and C++
- Support for large-scale computations (100M+ iterations)

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
python-dotenv>=0.21.0
gradio>=4.0.0
ipython>=8.0.0
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
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
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

### Model Selection
```python
claude_model = 'claude-sonnet-4-20250514'
openai_model = 'gpt-4o-mini'
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
python langbridge.ipynb
# Or if using Jupyter
jupyter notebook langbridge.ipynb
```

#### 2. Using the Web UI
- **Input Python Code**: Paste your Python algorithm in the left textbox
- **Select Model**: Choose between GPT or Claude from dropdown
- **Convert**: Click "Convert code" button
- **Execute Python**: View Python execution results and timing
- **Execute C++**: View C++ compilation and execution results
- **Compare**: Analyze performance differences

### Programmatic Usage

#### Convert Code with GPT
```python
from langbridge import optimize_gpt

python_code = """
def calculate(n):
    return sum(i**2 for i in range(n))
"""

cpp_code = optimize_gpt(python_code)
print(cpp_code)
```

#### Convert Code with Claude
```python
from langbridge import optimize_claude

cpp_code = optimize_claude(python_code)
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

## 🤖 API Models

### OpenAI GPT-4o-mini
- **Model ID**: `gpt-4o-mini`
- **Use Case**: Fast code conversion with good optimization
- **Strengths**: Quick response, cost-effective
- **Context Window**: 128k tokens

### Anthropic Claude Sonnet 4
- **Model ID**: `claude-sonnet-4-20250514`
- **Use Case**: High-quality optimized C++ generation
- **Strengths**: Superior optimization, better type handling
- **Context Window**: 200k tokens

### System Prompt
```
You are an assistant that reimplements Python code in high performance C++ 
for an M4 Mac Pro. Respond only with C++ code; use comments sparingly and 
do not provide any explanation other than occasional comments. The C++ 
response needs to produce an identical output in the fastest possible time.
```

---

## 📊 Performance Benchmarks

### Example: Pi Calculation (100M iterations)

#### Python Execution
```
Result: 3.141592658589
Execution Time: 5.285664 seconds
```

#### C++ Execution (GPT-Optimized)
```
Result: 3.141592658589
Execution Time: 0.131795 seconds
```

#### C++ Execution (Claude-Optimized)
```
Result: 3.141592662604
Execution Time: 0.092037 seconds
```

**Performance Improvement**: **~50-57x faster** than Python

### Example: Maximum Subarray Sum (10K array, 20 runs)

#### Python Execution
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 15.477061 seconds
```

#### C++ Execution (GPT-Optimized)
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 0.001839 seconds
```

#### C++ Execution (Claude-Optimized)
```
Total Maximum Subarray Sum (20 runs): 10980
Execution Time: 0.487203 seconds
```

**Performance Improvement**: **~8400-31x faster** than Python

---

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│            Gradio Web Interface (UI Layer)          │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │  Python Editor   │      │   C++ Editor     │    │
│  └──────────────────┘      └──────────────────┘    │
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │ Python Executor  │      │  C++ Executor    │    │
│  └──────────────────┘      └──────────────────┘    │
├─────────────────────────────────────────────────────┤
│              Code Conversion Layer                  │
├────────────────┬────────────────┬──────────────────┤
│                │                │                  │
│   OpenAI API   │  Anthropic API │  System Prompts  │
│   (GPT-4o)     │  (Claude)      │                  │
│                │                │                  │
├────────────────┴────────────────┴──────────────────┤
│           Execution & Compilation Layer             │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │  Python Exec     │      │   C++ Compiler   │    │
│  │  (via exec())    │      │   (clang++)      │    │
│  └──────────────────┘      └──────────────────┘    │
│                                                       │
│  Output Capture (io.StringIO) & File Writing       │
├─────────────────────────────────────────────────────┤
```

### Key Functions

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `optimize()` | Route to appropriate AI model | Python code, model name | Generator yielding C++ code |
| `optimize_gpt()` | Convert using OpenAI | Python code | C++ code |
| `optimize_claude()` | Convert using Anthropic | Python code | C++ code |
| `execute_python()` | Run Python with output capture | Python code | Captured stdout |
| `execute_cpp()` | Compile and run C++ | C++ code | Executable output |
| `user_prompt_for()` | Build AI prompt | Python code | Formatted prompt string |
| `write_output()` | Save C++ to file | C++ code | File write confirmation |

---

## 💡 Examples

### Example 1: Simple Calculation

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

#### Python Input (Max Subarray Sum)
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

# Performance test with n=10000, 20 iterations
```

#### Key Optimizations Applied
- Linear congruential generator inlining
- Type upgrade to int64_t/int128 for overflow prevention
- Algorithm-level optimization (Kadane's variant)
- Vectorization hints for compiler

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

### Issue: C++ compilation fails
**Solution**: Check for syntax errors in generated code
```bash
clang++ -std=c++17 yourfile.cpp -o out  # Verbose compilation
```

### Issue: Python exec() returns None in Gradio
**Solution**: Use `io.StringIO` to capture stdout (already implemented in `execute_python()`)

### Issue: Memory issues with large arrays
**Solution**: Reduce array size or increase system RAM. Monitor with:
```bash
top -l 1 | grep Memory
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

---

## 📝 File Structure

```
langbridge/
├── langbridge.ipynb          # Main Jupyter notebook
├── .env                      # Environment variables (git-ignored)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── optimized.cpp             # Generated C++ files
└── LICENSE                   # MIT License
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support & Contact

- **Issues**: Open GitHub issues for bugs
- **Questions**: Create discussions in GitHub

---

## 🎓 Citation

If you use LangBridge in research, please cite:
```bibtex
@software{langbridge2025,
  author = Asutosha Nanda,
  title = {LangBridge},
  year = {2025},
  url = {https://github.com/AsutoshaNanda/LangBridge}
}
```

---

<div align="center">

**[⬆ Back to Top](#-langbridge-python-to-c-code-optimizer)**

Made with ❤️ for high-performance computing

</div> 
