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

- Run unit tests: `pytest`
