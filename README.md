# ML Pipeline Template

A comprehensive example of a well-architected machine learning pipeline using **Pydantic** for configuration validation and **ExCa** for experiment orchestration and caching. This project demonstrates best practices for building maintainable, reproducible ML systems.

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

#### 4. Install Dependencies (single file)

```bash
# Option A: Conda env (recommended)
conda create -n ml-pipeline-env python=3.10 -y
conda activate ml-pipeline-env
pip install -r requirements.txt

# Option B: Virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Development Workflow

This project uses a simple branch strategy with `main` and short‑lived `feature/*` branches.

#### 🌿 Branch Strategy

- `main`: Stable, production‑ready code
- `feature/<name>`: Individual feature branches created off `main`

#### 📝 Development Process (copy‑paste friendly)

1) Create a new feature branch from `main`:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2) Develop your changes:

- Implement the change.
- Adapt or add tests in the relevant places (under `tests/`).
- Update docs if behavior or CLI changes.

3) Run checks locally:

   ```bash
   # Run tests
   pytest -v

   # Format/lint/type
   black src tests
   isort src tests
   flake8 src tests
   mypy src
   ```

4) Stage and commit with a clear, formatted message (Conventional Commits):

   ```bash
   # Stage relevant changes
   git add src/ tests/ configs/ README.md

   # Example commit messages
   git commit -m "feat: add MNIST flattening in data pipeline" \
               -m "Adds flatten flag in data prep and pipeline wiring; updates/adds tests."
   # or
   git commit -m "fix: correct input shape validation for linear models"
   ```

5) Push your branch:

   ```bash
   git push -u origin feature/your-feature-name
   ```

6) Open a Pull Request in your Git web UI:

- Click the purple “Create PR” (from `feature/your-feature-name` → `main`).
- Title: concise summary of the change.
- Description: what you did, why, and what tests you added/updated.
- Ensure there are no merge conflicts.

7) Merge and clean up:

- Merge the PR and delete the remote branch (or wait for approval if preferred).
- Sync your local `main` and delete your local feature branch:

  ```bash
  git checkout main
  git pull origin main
  git branch -d feature/your-feature-name
  ```

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

## 🧹 Code Style and Formatting

This repo enforces Black formatting and isort import ordering locally via pytest. If files are not formatted/sorted, `pytest` will fail.

Recommended commands:

```bash
# Ensure environment is installed
pip install -r requirements.txt

# Format code (auto-fix)
black src tests
isort src tests

# Linting and type checks
flake8 src tests
mypy src

# Run tests (includes Black and isort checks via pytest)
pytest -v
```

Optional: enable pre-commit hooks for automatic formatting and linting on commit.

```bash
pre-commit install
# Run all hooks on the repo
pre-commit run --all-files
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

### Caching behavior

- Stages decorated with `infra.cached_stage(name)` are cached on disk using a hash of `{experiment, stage, context}`.
- The data preparation stage provides `_cache_context=self.config.model_dump()` so any config change invalidates the cache.
- You can influence the key in custom cached stages by passing `_cache_context=...`:

  ```python
  @infra.cached_stage("precompute_features")
  def precompute():
      return expensive_result

  # Hit cache next time when context matches
  res = precompute(_cache_context={"dataset": "mnist", "ver": 1})
  ```

- Non-picklable results are computed normally and skipped from caching (with a meta file recorded for observability).

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

## 📋 Getting Started Checklist

- [ ] Clone template and set new origin
- [ ] Install dependencies
- [ ] Run tests to verify setup: `pytest tests/`
- [ ] Run example experiment: `python -m src.cli --config configs/local.yaml`
- [ ] Create feature branch: `git checkout -b feature/my-first-feature`
- [ ] Make a small change and verify CI passes
- [ ] Read through the codebase to understand the patterns

## 🆘 Troubleshooting

### Common Issues

**Import errors**: Ensure you installed from the unified file: `pip install -r requirements.txt`

**Test failures**: Check that all dependencies are installed: `pip install -r requirements.txt`

**Config validation errors**: Check YAML syntax and required fields in `schemas.py`

**Cache issues**: Delete `cache/` directory to reset: `rm -rf cache/`

## 📖 Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [ExCa Documentation](https://github.com/fairinternal/exca)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)

---

## 🧭 Using This Template For Your Own Project

This template is designed to be adapted quickly to your use case while preserving a clean, reproducible workflow.

### 1) Clone and re-point the repo

1. Clone this repository and rename the folder to your project name
2. Set a new GitHub remote to your own repository (see “Getting Started → Set New Origin” above)

### 2) Environment and tooling

- Use Python 3.10+
- Install dependencies (formatting, linting, tests included via single file):

```bash
pip install -r requirements.txt
pre-commit install  # Auto-run black + isort on each commit
pytest -v           # Verify everything is green locally
```

Notes:

- Local tests enforce Black and isort; CI runs the same checks on push/PR
- Integration tests are gated after unit/lint checks in CI

### 3) Understand the core components

- `src/experiment/schemas.py`: Pydantic models validate configuration (data/model/training/evaluation/experiment/logging). Start here to define your config surface.
- `src/experiment/data.py`: Data loading, transforms, splits, and `prepare_data`. Replace MNIST with your data and adapt `create_transforms`, `create_datasets`, and `create_dataloaders`.
- `src/experiment/models.py`: Model factories (linear/MLP/CNN examples). Add your architectures and extend the factory in `create_model`.
- `src/experiment/pipeline.py`: Orchestrates end‑to‑end flow (prepare → create model → train → evaluate → save artifacts). Minimal changes usually needed once data/models are wired.
- `src/experiment/infra.py`: ExCa integration for caching artifacts/metrics and stage wrappers.

### 4) Adapt the configuration

1. Copy or create a config under `configs/` (e.g., `configs/my_experiment.yaml`)
2. Update fields under `data`, `model`, and `training` to match your use case
3. Validate quickly:

```bash
python -m src.cli validate configs/my_experiment.yaml
```

### 5) Replace the dataset (quick path)

In `src/experiment/data.py`:

- Swap MNIST references for your dataset (e.g., custom `Dataset` implementation)
- Adjust transforms and shapes
- Ensure `validate_data_shape` reflects what your model expects

Run the data-only integration test locally by invoking:

```bash
pytest tests/test_integration.py::TestDataPipelineIntegration -v
```

### 6) Extend/add models

1. In `schemas.py`, add a new Pydantic model config (e.g., `MyModel`)
2. In `models.py`, implement a creator (e.g., `create_my_model`)
3. Add your model type to the factory in `create_model`

Smoke-test model creation:

```bash
pytest tests/test_models.py -v
```

### 7) Run the full pipeline

```bash
python -m src.cli run configs/my_experiment.yaml
```

Artifacts and metrics are written under `cache/` and `results/`. The pipeline logs key events and saves `training_history` and final metrics for traceability.

### 8) Keep quality gates green

- Auto-format runs on commit (pre-commit)
- Local tests enforce formatting/import order: `pytest -v`
- CI (GitHub Actions) runs: lint, black/isort checks, tests with coverage, and staged integration tests

### 9) Common adaptation checklist

- Update `schemas.py` to reflect your configuration needs
- Replace dataset logic in `data.py`
- Add or adapt models in `models.py`
- Tweak training hyperparameters via config (not code) where possible
- Keep tests green; add unit tests for new components and small integration tests for new workflows

With these pieces in place, you can scale from a simple experiment to a robust, reproducible pipeline without sacrificing clarity or speed.

---

Made by **Itay Inbar**
