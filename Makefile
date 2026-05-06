.PHONY: setup ingest serve ui test eval coverage lint docker-up docker-down clean

PYTHON ?= python
PIP ?= pip

setup:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

ingest:
	$(PYTHON) scripts/run_ingest.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/streamlit_app.py

test:
	pytest tests -m "not slow"

eval:
	pytest tests/eval -m slow

coverage:
	pytest tests -m "not slow" --cov=src --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .
	mypy src api scripts

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('data/vectorstore', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('.mypy_cache', ignore_errors=True); shutil.rmtree('.ruff_cache', ignore_errors=True)"
