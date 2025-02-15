# Configuration Guide

## YAML Configuration

YAMLLM uses YAML files for configuration. Here's a complete reference of available options:

```yaml
# Model Settings
model: gpt-4-turbo-preview
temperature: 0.7
max_tokens: 500

# System Configuration
system_prompt: "You are a helpful AI assistant."
retry_attempts: 3
timeout: 30

# Memory Settings
memory:
  type: "vector"
  capacity: 1000
```