# Examples

## Basic Usage

### Simple Query
```python
from yamllm.core.llm import GoogleGemini
import os
import dotenv

dotenv.load_dotenv()

config_path = "config.yaml"
llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

# Make a query - response printing is handled automatically
response = llm.query("Give me some boilerplate pytorch code please")
```

### Interactive Chat
```python
from yamllm.core.llm import GoogleGemini
from rich.console import Console
import os
import dotenv

dotenv.load_dotenv()
console = Console()

config_path = "config.yaml"
llm = GoogleGemini(config_path=config_path, api_key=os.environ.get("GOOGLE_API_KEY"))

while True:
    try:          
        prompt = input("\nHuman: ")
        if prompt.lower() == "exit":
            break
        
        response = llm.query(prompt)
        if response is None:
            continue
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
    except ValueError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
    except Exception as e:
        console.print(f"[red]An error occurred:[/red] {str(e)}")
```

## Provider Examples

### OpenAI
```python
from yamllm.core.llm import OpenAIGPT
import os
import dotenv

dotenv.load_dotenv()

config_path = "config.yaml"
llm = OpenAIGPT(config_path=config_path, api_key=os.environ.get("OPENAI_API_KEY"))
response = llm.query("What is machine learning?")
```

### Mistral AI
```python
from yamllm.core.llm import MistralAI
import os
import dotenv

dotenv.load_dotenv()

config_path = "config.yaml"
llm = MistralAI(config_path=config_path, api_key=os.environ.get("MISTRAL_API_KEY"))
response = llm.query("Explain quantum computing")
```

### DeepSeek
```python
from yamllm.core.llm import DeepSeek
import os
import dotenv

dotenv.load_dotenv()

config_path = "config.yaml"
llm = DeepSeek(config_path=config_path, api_key=os.environ.get("DEEPSEEK_API_KEY"))
response = llm.query("Help me write a Python function")
```

## Configuration Examples

### OpenAI Configuration
```yaml
provider:
  name: "openai"
  model: "gpt-4-turbo-preview"
  api_key: ${OPENAI_API_KEY}
  base_url: null  # optional: for custom endpoints

model_settings:
  temperature: 0.7
  max_tokens: 1000
```

### Google Configuration
```yaml
provider:
  name: "google"
  model: "gemini-pro"
  api_key: ${GOOGLE_API_KEY}
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

model_settings:
  temperature: 0.7
  max_tokens: 1000
```

### Mistral Configuration
```yaml
provider:
  name: "mistralai"
  model: "mistral-small-latest"
  api_key: ${MISTRAL_API_KEY}
  base_url: "https://api.mistral.ai/v1/"

model_settings:
  temperature: 0.7
  max_tokens: 1000
```

## Error Handling Examples

### Basic Error Handling
```python
from yamllm.core.llm import GoogleGemini
from rich.console import Console
import os
import dotenv

dotenv.load_dotenv()
console = Console()

try:
    llm = GoogleGemini(config_path="config.yaml", api_key=os.environ.get("GOOGLE_API_KEY"))
    response = llm.query("Your prompt here")
except FileNotFoundError as e:
    console.print(f"[red]Configuration file not found:[/red] {e}")
except ValueError as e:
    console.print(f"[red]Configuration error:[/red] {e}")
except Exception as e:
    console.print(f"[red]An error occurred:[/red] {str(e)}")
```

## Environment Setup

### Install Package
```bash
# Using pip
pip install yamllm-core

# Using uv
uv add yamllm-core
```

### Environment Variables
Create a `.env` file in your project root:
```plaintext
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
MISTRAL_API_KEY=your_api_key_here
DEEPSEEK_API_KEY=your_api_key_here
```

## Additional Resources

- Full documentation: [YAMLLM Documentation](https://github.com/codehalwell/yamllm/docs)
- Source code: [GitHub Repository](https://github.com/codehalwell/yamllm)
- Issue tracker: [GitHub Issues](https://github.com/codehalwell/yamllm/issues)