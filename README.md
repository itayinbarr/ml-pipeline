# Research Lab Project Template

A comprehensive template for reproducible machine learning research using **Pydantic** for configuration validation and **ExCa** for experiment orchestration and caching.

## 🎯 Features

- **Type-safe configurations** with Pydantic validation
- **Experiment reproducibility** with ExCa caching
- **Automated testing** with GitHub Actions CI/CD
- **MNIST classification example** with multiple model types
- **Clean project structure** following ML best practices
- **Comprehensive test coverage** for reliable development

## 📁 Project Structure

```
project-template/
├── .github/workflows/          # CI/CD automation
├── configs/                    # Experiment configurations
│   ├── local.yaml             # Development settings
│   └── production.yaml        # Full-scale training
├── data/
│   ├── raw/                   # Original datasets (git-ignored)
│   └── interim/               # Processed data
├── results/
│   ├── models/                # Saved model artifacts
│   ├── metrics/               # Performance metrics
│   └── plots/                 # Visualizations
├── src/experiment/            # Core implementation
│   ├── schemas.py            # Pydantic configuration models
│   ├── data.py               # Data loading & preprocessing
│   ├── models.py             # ML models (Linear, MLP, CNN)
│   ├── pipeline.py           # Experiment orchestration
│   └── infra.py              # ExCa infrastructure wrapper
├── tests/                     # Comprehensive test suite
└── notebooks/                 # Analysis notebooks
```

## 🚀 Getting Started

### Setting Up Your Project

This template is designed to be cloned and then connected to your own repository. Follow these steps:

#### 1. Clone This Template

```bash
# Clone the template repository
git clone https://github.com/yourusername/project-template.git MyProjectName
cd MyProjectName
```

#### 2. Create Your New Repository

1. Go to GitHub and create a new repository called `MyProjectName`
2. **Do not** initialize it with README, .gitignore, or license (we already have these)

#### 3. Set New Origin

```bash
# Remove connection to template repository
git remote rm origin

# Connect to your new repository
git remote add origin git@github.com:yourusername/MyProjectName.git

# Push to your new repository
git push -u origin main
```

#### 4. Install Dependencies

```bash

# using conda/mamba
conda env create -f environment.yml  # If you create one
```

### Development Workflow

This project follows a **branch-based development workflow**:

#### 🌿 Branch Strategy

- **`main`**: Production-ready, stable code
- **`dev`**: Integration branch for new features
- **`feature/*`**: Individual feature development

#### 📝 Development Process

1. **Create a feature branch** from `dev`:

   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **Develop your feature**:

   - Write code following the existing patterns
   - Add tests for new functionality
   - Update documentation as needed

3. **Test locally**:

   ```bash
   # Run tests
   pytest tests/ -v

   # Check code formatting
   black --check src tests
   isort --check-only src tests

   # Type checking
   mypy src
   ```

4. **Push and create PR**:

   ```bash
   git push origin feature/your-feature-name
   ```

   - Create a Pull Request to `dev` branch
   - Wait for CI tests to pass
   - Request code review

5. **Merge to main**:
   - After review, merge `dev` → `main`
   - CI will run full integration tests

## 🔧 Usage

### Running Experiments

The project provides a simple CLI interface:

```bash
# Run experiment with default config
python -m src.cli --config configs/local.yaml

# Run with custom config
python -m src.cli --config configs/production.yaml

# Get help
python -m src.cli --help
```

### Configuration System

Experiments are configured using YAML files validated by Pydantic:

```yaml
# configs/local.yaml
data:
  dataset: mnist
  batch_size: 32
  validation_split: 0.1

model:
  name: mlp # or 'linear', 'cnn'
  hidden_size: 128
  dropout: 0.1

training:
  epochs: 10
  learning_rate: 0.001
  optimizer: adam

experiment:
  name: "mnist_baseline"
  random_seed: 42
  cache_dir: cache
```

### Available Models

The template includes three model types:

1. **Linear Model**: Simple logistic regression

   ```yaml
   model:
     name: linear
     input_size: 784
     num_classes: 10
   ```

2. **Multi-Layer Perceptron**:

   ```yaml
   model:
     name: mlp
     hidden_size: 128
     num_layers: 2
     dropout: 0.1
   ```

3. **Convolutional Neural Network**:
   ```yaml
   model:
     name: cnn
     channels: [32, 64]
     kernel_size: 3
     dropout: 0.2
   ```

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test complete workflows end-to-end
- **Schema tests**: Validate Pydantic configuration models
- **Pipeline tests**: Test experiment orchestration

## 📊 Example: MNIST Classification

The template includes a complete MNIST classification example:

```python
from src.experiment import ExperimentConfig, Experiment
import yaml

# Load configuration
with open('configs/local.yaml') as f:
    config_dict = yaml.safe_load(f)

# Create and validate config
config = ExperimentConfig(**config_dict)

# Run experiment
experiment = Experiment.from_config(config)
results = experiment.run()

print(f"Final accuracy: {results['accuracy']:.3f}")
print(f"Results saved to: {results['output_path']}")
```

## 🔄 ExCa Integration

The template uses ExCa for caching and reproducibility:

- **Data loading**: Cached to avoid re-downloading
- **Feature processing**: Cached preprocessed features
- **Model training**: Cached trained models
- **Evaluation**: Cached metrics and predictions

Cache artifacts are stored in `cache/` (git-ignored) with full provenance tracking.

## 📈 Continuous Integration

GitHub Actions automatically:

✅ **On every push/PR**:

- Run all tests across Python versions
- Check code formatting (black, isort)
- Lint code (flake8)
- Type checking (mypy)
- Test configuration validation

✅ **On main branch**:

- Run integration tests
- Build documentation
- Upload coverage reports

## 📚 Adding New Features

### Adding a New Model

1. **Define the model config** in `src/experiment/schemas.py`:

   ```python
   class MyModel(BaseModel):
       name: Literal["my_model"] = "my_model"
       param1: float = 1.0
       param2: int = 100
   ```

2. **Add to the union**:

   ```python
   ModelConfig = Union[LinearModel, MLPModel, CNNModel, MyModel]
   ```

3. **Implement the model** in `src/experiment/models.py`:

   ```python
   def create_my_model(config: MyModel) -> torch.nn.Module:
       # Implementation here
       pass
   ```

4. **Update the factory**:

   ```python
   def create_model(config: ModelConfig) -> torch.nn.Module:
       if isinstance(config, MyModel):
           return create_my_model(config)
       # ... other cases
   ```

5. **Add tests** in `tests/test_models.py`

### Adding New Data Sources

Follow the same pattern in `src/experiment/data.py` and update the schemas accordingly.

## 🤝 Code Review Guidelines

For easy code review:

1. **Check CI status**: All tests must pass
2. **Review config changes**: YAML diffs show experiment modifications clearly
3. **Check test coverage**: New code should include tests
4. **Validate schemas**: Pydantic catches config errors early
5. **Review caching**: ExCa ensures reproducible results

## 📋 Checklist for New Students

- [ ] Clone template and set new origin
- [ ] Install dependencies
- [ ] Run tests to verify setup: `pytest tests/`
- [ ] Run example experiment: `python -m src.cli --config configs/local.yaml`
- [ ] Create feature branch: `git checkout -b feature/my-first-feature`
- [ ] Make a small change and verify CI passes
- [ ] Read through the codebase to understand the patterns

## 🆘 Troubleshooting

### Common Issues

**Import errors**: Ensure you installed with `-e` flag: `pip install -e .[dev]`

**Test failures**: Check that all dependencies are installed: `pip install -e .[dev]`

**Config validation errors**: Check YAML syntax and required fields in `schemas.py`

**Cache issues**: Delete `cache/` directory to reset: `rm -rf cache/`

## 📖 Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [ExCa Documentation](https://github.com/fairinternal/exca)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)

---

Made by **Itay Inbar**
