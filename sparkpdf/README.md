

## Development

### Setup

```bash
  git clone
  cd spark-pdf
```

### Install dependencies

```bash
  poetry install
```

### Run tests

```bash
  poetry run pytest --cov=sparkpdf --cov-report=html:coverage_report tests/ 
```

### Build package

```bash
  poetry build
```

### Build documentation

```bash
  poetry run sphinx-build -M html source build
```

### Release

```bash
  poetry version patch
  poetry publish --build
```
