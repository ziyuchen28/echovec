BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= RelWithDebInfo

.PHONY: all configure build test bench clean distclean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

bench: build
	./$(BUILD_DIR)/bench_dot --dim 1536 --iters 200000 --warmup 1000

clean:
	cmake --build $(BUILD_DIR) --target clean

distclean:
	rm -rf $(BUILD_DIR)
