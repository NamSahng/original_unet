PYTHON=3.8
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install --file requirements.txt -c conda-forge -c pytorch

format:
	black .
	isort .

lint:
	pytest src --flake8 --pylint --mypy
