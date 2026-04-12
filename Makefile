.PHONY: install-dev test-all docs docs-serve bench-all clean

SUBPROJECTS := trnfft trnblas trnrand trnsolve trnsparse trntensor

install-dev:
	@for p in $(SUBPROJECTS); do \
		echo "==> pip install -e ./$$p"; \
		pip install -e "./$$p[dev]" || exit 1; \
	done
	pip install -e ".[dev]"

test-all:
	@for p in $(SUBPROJECTS); do \
		echo "==> pytest $$p"; \
		( cd $$p && pytest tests/ -v -x --tb=short -m "not neuron" ) || exit 1; \
	done
	pytest tests/ -v -x --tb=short -m "not neuron"

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

bench-all:
	@for p in $(SUBPROJECTS); do \
		if [ -d "$$p/benchmarks" ]; then \
			echo "==> benchmarks in $$p"; \
			( cd $$p && pytest benchmarks/ --benchmark-only ) || true; \
		fi; \
	done

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf site/ build/ dist/
