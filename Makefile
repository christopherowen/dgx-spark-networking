.PHONY: test simulate compile nccl clean

test:
	python3 -m unittest discover -s tests

simulate:
	mkdir -p build && $(CC) -O1 -g -std=gnu11 -Wall -Wextra -Werror -fsanitize=address,undefined -pthread \
	  -Itests/oneshot_sim tests/oneshot_sim/simulate.c -o build/simulate && build/simulate

compile:
	python3 -m compileall -q sparknet tests benchmarks

nccl:
	scripts/build-nccl.sh build/nccl

clean:
	rm -rf build .work
