# Contributing

Thank you for your interest in this project! This is a personal
academic project, but contributions and suggestions are welcome.

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## Development Setup

```bash
# Clone
git clone https://github.com/<your-username>/phone-defect-detection.git
cd phone-defect-detection

# Backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend: open frontend/ in WeChat DevTools
```

## Code Style

- **Python**: Follow PEP 8. Use type hints and docstrings.
- **JavaScript**: Follow the existing style in the mini program.
- **Documentation**: Write in English for code and docs; Chinese is
  acceptable for thesis-related materials.

## Reporting Issues

When reporting bugs, please include:
- Python / FastAPI / ONNX Runtime versions
- Steps to reproduce
- Expected vs. actual behavior
- Error messages or stack traces

## Questions?

Feel free to open an issue or contact the author.
