.PHONY: clean clean-pyc clean-test clean-build help

help:
	@echo "clean        - remove all build, test, coverage and Python artifacts"
	@echo "clean-build  - remove build artifacts"
	@echo "clean-pyc    - remove Python file artifacts"
	@echo "clean-test   - remove test and coverage artifacts"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -rf build/
	rm -rf dist/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-test:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -name '.pytest_cache' -exec rm -rf {} +
