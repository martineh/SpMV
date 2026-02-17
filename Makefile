include Makefile.inc

CC = g++

ifeq ($(SIMD), AVX2)
	CFLAGS=-mavx2 -mfma -march=haswell -maes
else ifeq ($(SIMD), NEON)
	CFLAGS=-march=armv8-a
else ifeq ($(SIMD), RVV1_M2_256)
	CFLAGS=-march=rv64gcv_zfh 
else ifeq ($(SIMD), SVE_128)
	CFLAGS=-march=armv8-a+sve
else ifeq ($(SIMD), SVE_256)
	CFLAGS=-march=armv8-a+sve2
else
	$(error ERROR: Type of arch='$(SIMD) unsuported.)
endif

CFLAGS += -Wall -O3 -fopenmp -D$(SIMD)
LDFLAGS = -lm -fopenmp

TARGET  = build/spmv
SRC_DIR = src
OBJ_DIR = build

SRCS := $(wildcard $(SRC_DIR)/*c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

ifneq ("$(GO_HIGHWAY_HOME)", "")
    NEW_SRCS := $(wildcard $(SRC_DIR)/go_highway/*.c)
    SRCS     += $(NEW_SRCS)
    OBJS     += $(NEW_SRCS:$(SRC_DIR)/go_highway/%.c=$(OBJ_DIR)/%.o)
    CFLAGS   += -I$(GO_HIGHWAY_HOME) -DGO_HIGHWAY
    LDFLAGS  += -L$(GO_HIGHWAY_HOME)/build -lhwy
endif

all: $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/go_highway/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

spmv: driver_spmv.o $(OBJ_DIR)
	$(CC) $(LDFLAGS) $^ -o $@

asm: csr_highway.s

csr_highway.s: src/go_highway/csr_highway.c
	$(CC) $(CFLAGS) -S -fverbose-asm -DGO_HIGHWAY $< -o $@

clean:
	rm -rf build/*
	rm -rf asm

