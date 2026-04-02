BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= RelWithDebInfo

DOT_IMPL ?= auto
# auto, avx2, scalar

.PHONY: all configure build test test-verbose bench-dot bench-search-flat clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-verbose: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -V

bench-dot: build
	./$(BUILD_DIR)/bench_dot --dim 1536 --iters 10000000 --warmup 10000 --impl $(DOT_IMPL)

bench-search-flat: build
	./$(BUILD_DIR)/bench_search_flat --count 20000 --dim 1536 --k 10 --iters 100 --warmup 3 --impl $(DOT_IMPL)

clean:
	rm -rf $(BUILD_DIR)
