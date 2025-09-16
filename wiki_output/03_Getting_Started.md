## Getting Started
### Prerequisites
- **System Requirements**
  - Python 3.9 or higher
  - Basic understanding of Python and command-line usage

- **Required Software and Tools**
  - Python package manager (`pip`)
  - Virtual environment manager (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aanshshah/llm_token_analytics_lib.git
   cd llm_token_analytics_lib
   ```

2. Install the library and dependencies:
   ```bash
   pip install -e .
   ```

3. (Optional) Install provider dependencies for data collection:
   ```bash
   pip install llm-token-analytics[providers]
   ```

4. Set up your environment variables for API keys (if applicable):
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export GOOGLE_CLOUD_PROJECT="your-project"
   ```

### Verification Steps
To verify the installation, run the basic simulation example:
```bash
python examples/01_basic_simulation.py
```
