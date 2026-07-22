# ExperimentGuard developer shortcuts.
# Assumes a virtualenv is active (python -m venv .venv && source .venv/bin/activate).

.PHONY: help install test data demo api docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run the full test suite
	pytest -q

data:  ## Generate the three sample scenarios into data/
	python scripts/generate_sample_data.py

demo: data  ## Run the CLI on all three scenarios
	@for f in winner srm guardrail_fail; do \
		echo "===== $$f ====="; \
		python run_analysis.py data/$$f.csv 2>/dev/null | grep -A2 "Decision:"; \
		echo; \
	done

report: data  ## Write the winner HTML report to data/report_winner.html
	python run_analysis.py data/winner.csv --html data/report_winner.html >/dev/null

api:  ## Serve the FastAPI app (http://localhost:8000/docs)
	uvicorn api:app --reload

docker:  ## Build and run the container on port 8000
	docker build -t experimentguard .
	docker run --rm -p 8000:8000 experimentguard

clean:  ## Remove caches and generated data
	rm -rf .pytest_cache **/__pycache__ data/*.csv data/*.html
