.PHONY: help install app run serve test unit validation performance bench lint docs clean

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

help:
	@echo "install      create the virtual environment and install Materia"
	@echo "run          open the native desktop window"
	@echo "serve        run the core with the interface on a local port"
	@echo "app          build the macOS application bundle"
	@echo "test         run every suite"
	@echo "unit         implementation tests only"
	@echo "validation   physical validation cases only"
	@echo "performance  scaling guards only"
	@echo "bench        measured benchmarks for this machine"
	@echo "clean        remove build artefacts and caches"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"
	$(BIN)/pip install pywebview
	@echo "done: run 'make run'"

run:
	$(BIN)/materia

serve:
	$(BIN)/materia serve --open

app:
	$(BIN)/python tools/make_macos_app.py

test:
	$(BIN)/pytest -q

unit:
	$(BIN)/pytest tests/unit tests/integration -q

validation:
	$(BIN)/pytest -m validation -q

performance:
	$(BIN)/pytest -m performance -q

bench:
	$(BIN)/materia bench

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
