.PHONY: run clean install test lint format

run:
	pipenv run python src

clean:
	rm -r build/ dist/ **/*.egg-info/ .mypy_cache/ .pytest_cache && find . -name __pycache__ -type d -exec rm -r {} \;

install:
	pipenv install

test:
	pipenv run pytest test && make lint

lint:
	make type-check && make format && pipenv run flake8 src test

format:
	pipenv run black src test && pipenv run isort src test --profile black

type-check:
	pipenv run mypy src test
