.PHONY: install lint fmt typecheck test eval bench serve demo train docker

install:        ## install runtime + dev dependencies
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -q -m "not slow"

test-all:
	uv run pytest -q

eval:           ## offline eval suite with the reference investigator, gated against baseline
	uv run sigil eval --model reference --seeds 3 --gate

bench:
	uv run sigil bench

serve:
	uv run sigil serve

demo:
	uv run sigil demo

train:          ## retrain the CDP model (needs the `train` extra: uv sync --extra train)
	uv run --extra train python -m training.train_cdp

docker:
	docker build -t sigil:latest .
