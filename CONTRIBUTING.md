# Contributing to Lo-Fi Music Generator

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- FluidSynth (optional, for audio synthesis)
- CUDA-capable GPU (optional, for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lofi-generator.git
   cd lofi-generator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Quality Standards

### Code Formatting

We use several tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security scanning

Format your code before committing:

```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

### Pre-commit Hooks

Pre-commit hooks will automatically run when you commit. If they fail, fix the issues and commit again.

To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Mark tests appropriately: `@pytest.mark.unit`, `@pytest.mark.integration`
- Aim for 80%+ code coverage
- Use fixtures from `conftest.py`

Example:

```python
import pytest
from src.tokenizer import LoFiTokenizer

@pytest.mark.unit
def test_tokenizer_initialization(test_config):
    tokenizer = LoFiTokenizer(test_config)
    assert tokenizer.vocab_size > 0
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Provide a clear description
   - Reference any related issues
   - Ensure all CI checks pass

### Pull Request Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts with main branch

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def generate_track(
    self,
    tempo: Optional[float] = None,
    key: Optional[str] = None,
    mood: Optional[str] = None,
) -> Tuple[List[int], Dict]:
    """Generate a single lo-fi track.

    Args:
        tempo: Target tempo in BPM (random if None)
        key: Target musical key (random if None)
        mood: Target mood (random if None)

    Returns:
        Tuple of (generated_tokens, metadata)

    Raises:
        GenerationError: If generation fails

    Example:
        >>> tokens, metadata = generator.generate_track(tempo=75)
    """
```

### Building Documentation

```bash
cd docs
sphinx-build -b html . _build/html
```

## Docker Development

### Using Docker

```bash
# Build development image
docker-compose build dev

# Run development container
docker-compose run --rm dev bash

# Run tests in container
docker-compose run --rm dev pytest

# Run Jupyter notebook
docker-compose up jupyter
```

## Continuous Integration

Our CI pipeline runs on every pull request:

- **Linting** (flake8, black, isort)
- **Type checking** (mypy)
- **Tests** (pytest on multiple Python versions)
- **Security scanning** (bandit)
- **Code coverage** (codecov)

Ensure all checks pass before requesting review.

## Release Process

Releases are automated via GitHub Actions:

1. Update version in `pyproject.toml`
2. Create a tag: `git tag v0.1.0`
3. Push tag: `git push origin v0.1.0`
4. GitHub Actions will build and publish

## Getting Help

- **Issues:** Report bugs or request features via GitHub Issues
- **Discussions:** Ask questions in GitHub Discussions
- **Documentation:** Check the docs at [link]

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something great together!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
