CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Werror -Iinclude
FIXTURES = $(wildcard tests/fixtures/qmat_*.bin)

build/test_qmat: src/qmat.c tests/test_qmat.c include/qmat.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/qmat.c tests/test_qmat.c -o $@ -lm

test: build/test_qmat
	./build/test_qmat $(FIXTURES)

clean:
	rm -rf build

.PHONY: test clean
