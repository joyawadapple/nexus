.PHONY: install run test lint format

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .
