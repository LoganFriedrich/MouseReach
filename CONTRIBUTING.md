# Contributing to MouseReach

Thank you for your interest in contributing to MouseReach!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/LoganFriedrich/MouseReach.git
   cd MouseReach
   ```

2. Create a conda environment:
   ```bash
   conda create -n mousereach python=3.10
   conda activate mousereach
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

4. Verify installation:
   ```bash
   mousereach --help
   ```

## DeepLabCut Model

MouseReach requires a trained DeepLabCut model for pose estimation. You must train your own model on your specific camera setup and animal subjects. See the [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/intro.html) for training instructions.

## Code Style

- Follow PEP 8 guidelines
- Use type hints where practical
- Add docstrings to public functions
- Keep functions focused and testable

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests if available
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages

## Questions?

Open an issue on GitHub for questions or discussion.
