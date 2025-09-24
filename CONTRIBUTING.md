# Contributing

## Testing

- Run the default headless test suite (skips GUI/integration tests by default):
  `KOE_HEADLESS=1 PYTHONPATH=src pytest`
- To exercise GUI or integration tests explicitly:
  `PYTHONPATH=src pytest -m "gui or integration"`
