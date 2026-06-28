# pyTVC C++ backend Makefile
#
# Typical WSL/Linux usage from the repository root:
#   make -C cpp
#   make -C cpp test
#   make -C cpp python
#   make -C cpp embedded-check
#
# This Makefile intentionally builds the host-only Python binding layer outside
# the embedded core. The embedded core remains header-only and no-heap.

CXX ?= c++
PYTHON ?= python3
ARM_CXX ?= arm-none-eabi-g++

BUILD_DIR ?= build
INCLUDE_DIR := include
REPO_ROOT := ..
PY_PACKAGE_DIR := $(REPO_ROOT)/pytvc

CXXSTD ?= -std=c++17
WARNINGS ?= -Wall -Wextra -Wpedantic
OPT ?= -O2
PIC ?= -fPIC

CORE_CXXFLAGS := $(CXXSTD) $(WARNINGS) $(OPT) -I$(INCLUDE_DIR)
PY_CXXFLAGS := $(CORE_CXXFLAGS) $(PIC)

CORE_TEST_SRC := tests/core_tests.cpp
EMBEDDED_SMOKE_SRC := tests/embedded_compile.cpp
BINDINGS_SRC := bindings/bindings.cpp
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
CORE_HEADERS := $(call rwildcard,$(INCLUDE_DIR)/,*.hpp)

CORE_TEST_EXE := $(BUILD_DIR)/pytvc_core_tests
EMBEDDED_SMOKE_OBJ := $(BUILD_DIR)/embedded_compile.o

PY_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null)
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
PY_EXTENSION := $(PY_PACKAGE_DIR)/_pytvc_cpp$(PY_EXT_SUFFIX)

.PHONY: all core test python embedded-check clean dirs check-pybind11

all: core python

dirs:
	mkdir -p $(BUILD_DIR)

core: $(CORE_TEST_EXE)

$(CORE_TEST_EXE): $(CORE_TEST_SRC) $(CORE_HEADERS) | dirs
	$(CXX) $(CORE_CXXFLAGS) $< -o $@

test: $(CORE_TEST_EXE)
	./$(CORE_TEST_EXE)

check-pybind11:
	@$(PYTHON) -c "import pybind11" >/dev/null 2>&1 || \
		( echo "pybind11 is required. Install with: $(PYTHON) -m pip install pybind11" && exit 1 )
	@test -n "$(PY_EXT_SUFFIX)" || \
		( echo "$(PYTHON)-config did not provide an extension suffix" && exit 1 )

python: check-pybind11 $(PY_EXTENSION)

$(PY_EXTENSION): $(BINDINGS_SRC) $(CORE_HEADERS) | dirs
	$(CXX) $(PY_CXXFLAGS) $(PYBIND11_INCLUDES) -shared $< -o $@

embedded-check: $(EMBEDDED_SMOKE_OBJ)

$(EMBEDDED_SMOKE_OBJ): $(EMBEDDED_SMOKE_SRC) $(CORE_HEADERS) | dirs
	$(ARM_CXX) $(CXXSTD) -I$(INCLUDE_DIR) -DPYTVC_SCALAR_DOUBLE=0 \
		-mcpu=cortex-m33 -mthumb -ffreestanding -fno-exceptions -fno-rtti \
		-c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PY_PACKAGE_DIR)/_pytvc_cpp*.so
