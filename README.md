# Basic Deep Learning MNIST

From-scratch NumPy neural network for MNIST digit classification, plus Black, Ruff, and MyPy for code quality.

## Tooling

This boilerplate uses the following tools to maintain code quality:

### Black

[Black](https://github.com/psf/black) is the uncompromising Python code formatter. It enforces a strict and consistent code style, which helps to eliminate debates over formatting and makes code review faster. By using Black, you ensure that the code looks the same regardless of who wrote it.

### Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter, written in Rust. It can check for a wide range of errors and style issues, from simple syntax mistakes to complex bugs. It also includes a code formatter. In this boilerplate, it is used for linting and fixing common issues automatically.

### MyPy

[MyPy](http://mypy-lang.org/) is a static type checker for Python. It helps you write cleaner, more robust code by adding type hints to your functions and variables. MyPy can catch type-related errors before you even run your code, which can save you a lot of time debugging.

## Automation

This project uses `make` and `invoke` to automate common development tasks.

### Makefile

A `Makefile` is included to provide a simple and familiar interface for common commands. You can see the available commands by running `make help`.

### Invoke & tasks.py

[Invoke](https://www.pyinvoke.org/) is a Python task execution tool. The `tasks.py` file defines a set of tasks that can be run from the command line. This allows for more complex logic than a `Makefile` and is written in Python. You can see the available tasks by running `invoke --list`.

## Train the neural network

Install dependencies first (`invoke install` or `make install`). Run every command from the project root. The package lives under `src/`, so Python needs `PYTHONPATH=src` (the Invoke and Make targets set this for you).

All of the commands below run `python -m mnistnn`, which calls `train_mnist()` with these defaults:

- 15 epochs
- batch size 64
- learning rate 0.05 (decayed by 0.9 after each epoch)
- SGD momentum 0.9
- pixel values scaled to `[-0.5, 0.5]`
- sample plots enabled (`visualize=True`)

Training progress is written to `log/mnist-training.log`.

### Invoke

Works the same in bash, PowerShell, and cmd:

```text
invoke train
```

### Make

Use this in bash, Git Bash, or WSL. The Makefile uses Unix `PYTHONPATH=src python -m mnistnn`, so it is not for stock Windows cmd or PowerShell unless `make` is installed.

```bash
make train
```

### Python interpreter

Set `PYTHONPATH` in the current shell, then run the module.

bash, Git Bash, or WSL:

```bash
export PYTHONPATH=src
python -m mnistnn
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m mnistnn
```

cmd:

```bat
set PYTHONPATH=src
python -m mnistnn
```

To train without plot windows, set `PYTHONPATH` as above, then:

```text
python -c "from mnistnn.train_mnist import train_mnist; train_mnist(visualize=False)"
```

## Available Commands

You can use either `make` or `invoke` to run the following commands:

| Command | Makefile | Invoke | Description |
|---|---|---|---|
| **Train Network** | `make train` | `invoke train` | Trains the MNIST network (`python -m mnistnn`). |
| **Format Code** | `make format` | `invoke format` | Formats the code using Black. |
| **Check Formatting** | `make format-check` | `invoke format-check` | Checks if the code is formatted correctly without making changes. |
| **Lint Code** | `make lint` | `invoke lint` | Lints the code with Ruff to find potential issues. |
| **Lint and Fix** | `make lint-fix` | `invoke lint-fix` | Lints the code with Ruff and automatically fixes any issues it can. |
| **Type Check** | `make typecheck` | `invoke typecheck` | Runs MyPy to check for type errors. |
| **Quality Assurance** | `make qa` | `invoke qa` | Runs all the quality checks: format, lint-fix, and typecheck. |
| **Install Dependencies**| `make install` | `invoke install` | Installs the required Python packages from `requirements.txt`. |
| **Clean Project** | `make clean` | `invoke clean` | Removes cache directories and temporary files. |
| **Help** | `make help` | `invoke help` | Shows the list of available commands. |
