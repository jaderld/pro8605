run-api:
	uvicorn api.main:app --reload

run-frontend:
	streamlit run frontend/main.py

test:
	python -m unittest discover -s tests/unit

lint:
	flake8 src/ api/ frontend/

clean:
	rm -rf __pycache__ venv .pytest_cache .mypy_cache
