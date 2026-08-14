# Convenience wrapper for people (and cluster login nodes) without CMake.
#
# Scoped to this example only -- run it from
# examples/hybrid-object-matching/, not from the repository root. The lab's
# primary build is CMake at the root; this exists because a login node with no
# CMake is a real situation and this workload is the one that runs there.
#
# CMake remains the primary build -- it is what CI and the test suite use.
# This Makefile builds the same sources with the same feature detection so a
# `make` in a fresh checkout always produces a working binary.
#
#   make                 build with whatever MPI/OpenMP/CUDA is on PATH
#   make MPI=0 CUDA=0    force a plain multicore build
#   make test            build and run the unit tests
#   make bench           run the benchmark harness
#   make format          apply .clang-format
#   make clean

CC       ?= cc
NVCC     ?= nvcc
BUILD    ?= build
VERSION  ?= 1.0.0

WARNINGS := -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes
CFLAGS   ?= -O3 -std=c11
CPPFLAGS := -Iinclude -Isrc -DMSEARCH_VERSION='"$(VERSION)"'
LDLIBS   := -lm

# ---------------------------------------------------------------- detection
# Each feature is used when its toolchain is present and not explicitly
# disabled, so the same `make` works on a laptop and on a GPU node.
OPENMP ?= $(if $(shell $(CC) -fopenmp -E -x c /dev/null >/dev/null 2>&1 && echo yes),1,0)
MPI    ?= $(if $(shell command -v mpicc 2>/dev/null),1,0)
CUDA   ?= $(if $(shell command -v $(NVCC) 2>/dev/null),1,0)

SRCS := src/core/log.c src/core/problem.c src/core/parser.c src/core/writer.c \
        src/backends/registry.c src/backends/backend_serial.c \
        src/runtime/runner_local.c src/cli/options.c
CUDA_OBJS :=

ifeq ($(OPENMP),1)
  SRCS     += src/backends/backend_openmp.c
  CPPFLAGS += -DMSEARCH_HAVE_OPENMP
  CFLAGS   += -fopenmp
  LDFLAGS  += -fopenmp
endif

ifeq ($(MPI),1)
  CC        := mpicc
  SRCS      += src/runtime/runner_mpi.c
  CPPFLAGS  += -DMSEARCH_HAVE_MPI
  LINK      := mpicc
else
  LINK := $(CC)
endif

ifeq ($(CUDA),1)
  CPPFLAGS  += -DMSEARCH_HAVE_CUDA
  CUDA_OBJS := $(BUILD)/backend_cuda.o
  NVCCFLAGS ?= -O3 -std=c++14 -Iinclude -Isrc
  # nvcc links the CUDA runtime; the host linker needs it explicitly.
  LDLIBS    += -lcudart -lstdc++
  CUDA_LIBDIR ?= $(dir $(shell command -v $(NVCC)))../lib64
  LDFLAGS   += -L$(CUDA_LIBDIR)
endif

OBJS := $(patsubst %.c,$(BUILD)/%.o,$(SRCS))
TESTS := test_metric test_parser test_options test_backends

.PHONY: all test bench format clean help
all: $(BUILD)/msearch

$(BUILD)/msearch: $(OBJS) $(CUDA_OBJS) $(BUILD)/src/cli/main.o
	@mkdir -p $(@D)
	$(LINK) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	@echo "built $@  (openmp=$(OPENMP) mpi=$(MPI) cuda=$(CUDA))"

$(BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNINGS) -c $< -o $@

$(BUILD)/backend_cuda.o: src/backends/backend_cuda.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(filter -D%,$(CPPFLAGS)) -c $< -o $@

# Tests never link MPI: they exercise backends, and the MPI runtime has its own
# end-to-end test under CTest.
test: $(TESTS:%=$(BUILD)/%)
	@fail=0; for t in $(TESTS); do \
	  echo "== $$t"; ($(BUILD)/$$t) || fail=1; done; \
	exit $$fail

$(BUILD)/test_%: tests/test_%.c $(SRCS) $(CUDA_OBJS)
	@mkdir -p $(@D)
	$(LINK) $(CPPFLAGS) $(CFLAGS) $(WARNINGS) $(LDFLAGS) -Itests -o $@ $< $(SRCS) \
	  $(CUDA_OBJS) $(LDLIBS)

bench: $(BUILD)/msearch
	python3 bench/run_bench.py --binary $(BUILD)/msearch

format:
	@command -v clang-format >/dev/null || { echo "clang-format not found"; exit 1; }
	clang-format -i $(shell find src include tests -name '*.c' -o -name '*.h' -o -name '*.cu')

clean:
	rm -rf $(BUILD)

help:
	@sed -n '1,20p' Makefile
