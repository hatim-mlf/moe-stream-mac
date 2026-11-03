CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude
IO_CFLAGS = -O2 -std=gnu99 -Wall -Iinclude -Isrc/io -Ithird_party
FIXTURES = $(wildcard tests/fixtures/qmat_*.bin)

build/test_qmat: src/qmat.c src/qmat_row.c tests/test_qmat.c include/qmat.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/qmat.c src/qmat_row.c tests/test_qmat.c -o $@ -lm

build/test_ops: src/ops.c tests/test_ops.c include/ops.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/ops.c tests/test_ops.c -o $@ -lm

build/test_gdn: src/gdn.c tests/test_gdn.c include/gdn.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/gdn.c tests/test_gdn.c -o $@ -lm

test: build/test_qmat build/test_ops build/test_gdn
	./build/test_qmat $(FIXTURES)
	./build/test_ops tests/fixtures
	./build/test_gdn tests/fixtures

build/real_matvec: src/qmat.c src/io/k3_st.c tests/real_matvec.c
	@mkdir -p build
	$(CC) $(IO_CFLAGS) src/qmat.c src/io/k3_st.c tests/real_matvec.c -o $@ -lm

MODEL ?= $(HOME)/models/Qwen3.6-35B-A3B-4bit
CORE = src/qmat.c src/qmat_row.c src/ops.c src/gdn.c src/loader.c src/expert_store.c src/io/k3_st.c
build/test_gdn_layer: $(CORE) src/gdn_layer.c tests/test_gdn_layer.c include/gdn_layer.h
	@mkdir -p build
	$(CC) $(IO_CFLAGS) -Werror $(CORE) src/gdn_layer.c tests/test_gdn_layer.c -o $@ -lm

test-layer: build/test_gdn_layer
	../.venv/bin/python tests/gdn_layer_ref.py $(MODEL) 0 6 build/gdn_l0.bin
	./build/test_gdn_layer $(MODEL) 0 build/gdn_l0.bin
test-real: build/real_matvec
	../.venv/bin/python tests/check_real.py $(MODEL)

clean:
	rm -rf build

build/test_attn_layer: $(CORE) src/attn_layer.c tests/test_attn_layer.c include/attn_layer.h
	@mkdir -p build
	$(CC) $(IO_CFLAGS) -Werror $(CORE) src/attn_layer.c tests/test_attn_layer.c -o $@ -lm

test-attn: build/test_attn_layer
	../.venv/bin/python tests/attn_layer_ref.py $(MODEL) 3 7 build/attn_l3.bin
	./build/test_attn_layer $(MODEL) 3 build/attn_l3.bin

build/test_moe_layer: $(CORE) src/moe_layer.c tests/test_moe_layer.c include/moe_layer.h
	@mkdir -p build
	$(CC) $(IO_CFLAGS) -Werror $(CORE) src/moe_layer.c tests/test_moe_layer.c -o $@ -lm

test-moe: build/test_moe_layer
	../.venv/bin/python tests/moe_layer_ref.py $(MODEL) 0 6 build/moe_l0.bin
	./build/test_moe_layer $(MODEL) 0 build/moe_l0.bin

MODEL_SRC = $(CORE) src/gdn_layer.c src/attn_layer.c src/moe_layer.c src/config.c src/model.c
build/test_model: $(MODEL_SRC) tests/test_model.c include/model.h
	@mkdir -p build
	$(CC) $(IO_CFLAGS) -Werror $(MODEL_SRC) tests/test_model.c -o $@ -lm
build/run_model: $(MODEL_SRC) tests/run_model.c include/model.h
	@mkdir -p build
	$(CC) $(IO_CFLAGS) -Werror $(MODEL_SRC) tests/run_model.c -o $@ -lm

test-model: build/run_model
	../.venv/bin/python tests/model_check.py tokenize $(MODEL) build/prompt.i32
	./build/run_model $(MODEL) build/prompt.i32 12 build/engine_out.bin
	../.venv/bin/python tests/model_check.py check $(MODEL) build/engine_out.bin

.PHONY: test test-real test-layer test-attn test-moe test-model clean
