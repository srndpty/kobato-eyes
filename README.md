# kobato-eyes

kobato-eyes is a Windows-first desktop application for indexing local images with Danbooru tags and performing near-duplicate detection.

## Development Setup (Windows)

1. `python -m venv .venv`
2. `.venv\\Scripts\\activate`
3. `python -m pip install --upgrade pip`
4. `pip install -e .[dev]`
5. `pre-commit install`

## Running the Application

- Ensure the virtual environment is active.
- Launch the GUI: `python -m ui.app`

## Testing

- Run the default headless test suite (excludes GUI/integration):
  `KOE_HEADLESS=1 PYTHONPATH=src pytest`
- Include GUI and integration tests when running locally:
  `PYTHONPATH=src pytest -m "gui or integration"`

## Packaging

1. Ensure dependencies are installed: `pip install -e .[dev]`
2. Generate a Windows binary: `pyinstaller tools/kobato-eyes.spec`
3. Bundled artifacts are written to `dist/kobato-eyes/`.
4. The application stores user settings in `%APPDATA%\\kobato-eyes\\config.yaml`.

## Known Limitations

- The packaged ONNX Runtime defaults to CUDA; ship a CPU build if target systems lack NVIDIA GPUs.
- Place ONNX model weights alongside the executable before launching the tagging pipeline.
- Headless execution is not yet supported; the duplicate pipeline relies on the PyQt6 GUI event loop.

