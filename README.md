# mt-lr-morphology

**`mt-lr-morphology`** is a research codebase for a systematic literature review on the intersection of **Machine Translation (MT)**, **low-resource languages**, and **morphological complexity**.

This project explores how different morphological typologies (e.g., agglutinative, polysynthetic, fusional) impact MT performance and what techniques can address these challenges in low-resource settings.

## 🗂️ Project Structure

```
mt-lr-morphology/
├── data /              # folder storing the datasets for the project
├── gpt /               # the task package for the project
├── notebooks/          # notebooks for the project
├── pyproject.toml
```

## ⚡ Quickstart Guide

### 1. Install `uv` (Fast Package Manager)

We use `uv` for managing dependencies and environments. Follow the `uv` [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) for macOS, Linux, and Windows.

Install it via the following command (macOS/Linux):

Ensure it's in your `PATH`:

```bash
source $HOME/.local/bin/env
```

Verify `uv` installation:

```bash
uv --version
```

### 2. Create and Activate a Virtual Environment

Create a local virtual environment:

```bash
uv venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

### 3. Install the Project in Editable Mode

Install the `gpt` package in **_editable mode_**, which means changes to the Python files will be immediately available without needing to reinstall the package.

Install the `gpt` project and its dependencies into the virtual environment:

```bash
uv pip install -e .
```

This:

1. Installs all required packages from `requirements.txt`
2. Registers the `gpt` CLI for use. (e.g., the command below runs the `generate_answers.py` file)
   > `gpt categorization generate_answers`
3. Makes your local code changes immediately available without reinstallation

### 4. Set Up Pre-Commit Hooks

To ensure code quality and formatting consistency:

```bash
pre-commit install
```
