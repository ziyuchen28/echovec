BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= RelWithDebInfo

.PHONY: all configure build test test_verbose bench_scalar bench clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure


test_verbose: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -V


bench_scalar: build
	./$(BUILD_DIR)/bench_dot --dim 1536 --iters 10000000  --warmup 10000 --impl scalar 


bench: build
	./$(BUILD_DIR)/bench_dot --dim 1536 --iters 10000000 --warmup 10000 --impl auto


clean:
	rm -rf $(BUILD_DIR)
