# Contributing to YAMLLM

Thank you for your interest in contributing to YAMLLM! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/CodeHalwell/yamllm.git
```
3. Create a virtual environment:
```bash
cd yamllm
uv venv
```
4. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests:
```bash
pytest
```
4. Update documentation if needed
5. Commit your changes:
```bash
git add .
git commit -m "Description of changes"
```

### Pull Request Process

1. Push to your fork:
```bash
git push origin feature/your-feature-name
```
2. Create a Pull Request from your fork to our main repository
3. Wait for review and address any feedback

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Testing

- Write unit tests for new features
- Maintain test coverage above 80%
- Test both success and error cases

### Documentation

- Update relevant documentation
- Include docstrings
- Add example usage when appropriate

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, etc.)
- Reference issue numbers when applicable

## Getting Help

- Open an issue for bugs or feature requests
- Join our Discord community
- Check existing issues and documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.