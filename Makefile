.PHONY: help install lint test status demo shadow active sweep compress clean

help:
	@echo "npuhalo — AMD Strix Halo Heterogeneous NPU+iGPU Inference Suite"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    Install Python dependencies in editable mode"
	@echo "  make lint       Run static checks for syntax and undefined names"
	@echo "  make test       Run automated unit and regression tests (25 tests)"
	@echo "  make status     Check AMD XDNA 2 NPU device node, driver, and XRT health"
	@echo "  make demo       Run the interactive live stream verification terminal demo"
	@echo "  make shadow     Run 30-task agentic shadow autopsy (NPU verification logging)"
	@echo "  make sweep      Run NPU concurrency & capacity saturation benchmark"
	@echo "  make compress   Run NPU tool-output context compression evaluation"
	@echo "  make clean      Remove temporary cache and execution files"

install:
	pip install -e .

lint:
	ruff check .

test:
	python3 -m unittest discover -s tests -v

status:
	python3 scripts/npu_status.py

demo:
	python3 examples/demo_stream.py

shadow:
	python3 verifier/scripts/run_shadow.py

sweep:
	python3 verifier/scripts/sweep_concurrency.py

compress:
	python3 verifier/scripts/run_compressor.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build dist *.egg-info .pytest_cache
