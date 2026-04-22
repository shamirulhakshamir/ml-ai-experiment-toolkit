# Intercom AI Tooling POC

A proof-of-concept ML tooling framework demonstrating core capabilities for the
Senior Data Scientist - AI Tooling role at Intercom.

## Components

### 1. Model Evaluator (`src/model_evaluator.py`)
A comprehensive ML model evaluation framework that:
- Evaluates classification and regression models with standard metrics
- Compares multiple models side-by-side with ranked results
- Generates evaluation reports with threshold analysis
- Supports custom metric functions

### 2. Feature Selector (`src/feature_selector.py`)
An automated feature selection pipeline that:
- Computes variance threshold filtering to remove low-variance features
- Calculates mutual information scores for feature relevance
- Performs correlation-based redundancy removal
- Runs recursive feature elimination (RFE) with configurable estimators
- Combines multiple selection methods into a consensus ranking

### 3. Experiment Dashboard (`src/experiment_dashboard.py`)
An experiment tracking data layer that:
- Logs experiment runs with parameters, metrics, and metadata
- Compares experiments across runs with summary statistics
- Identifies best runs by a target metric
- Exports experiment data to JSON for downstream dashboards

## Setup

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
POC_Project/
  src/
    model_evaluator.py
    feature_selector.py
    experiment_dashboard.py
  tests/
    test_model_evaluator.py
    test_feature_selector.py
    test_experiment_dashboard.py
  requirements.txt
  README.md
```

## Author
Shamirul Hak Surbudeen
