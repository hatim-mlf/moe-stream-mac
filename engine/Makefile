CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude
IO_CFLAGS = -O2 -std=gnu99 -Wall -Iinclude -Isrc/io -Ithird_party
FIXTURES = $(wildcard tests/fixtures/qmat_*.bin)

build/test_qmat: src/qmat.c tests/test_qmat.c include/qmat.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/qmat.c tests/test_qmat.c -o $@ -lm

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
test-real: build/real_matvec
	../.venv/bin/python tests/check_real.py $(MODEL)

clean:
	rm -rf build

.PHONY: test test-real clean
