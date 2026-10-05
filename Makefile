.PHONY: test lint check simulate compile nccl clean

test:
	python3 -m unittest discover -s tests

lint:
	ruff check sparknet tests benchmarks examples

check: lint test
	git diff --check

simulate:
	mkdir -p build && $(CC) -O1 -g -std=gnu11 -Wall -Wextra -Werror -fsanitize=address,undefined -pthread \
	  -Itests/oneshot_sim tests/oneshot_sim/simulate.c -o build/simulate && build/simulate

compile:
	python3 -m compileall -q sparknet tests benchmarks examples

nccl:
	scripts/build-nccl.sh build/nccl

clean:
	rm -rf build .work
