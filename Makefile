CC = g++
ARCH=AVX2

ifeq ($(ARCH), AVX2)
	CFLAGS=-DAVX2 -mavx2 -mfma
else ifeq ($(ARCH), SVE)
	CFLAGS=-march=armv8-a+sve2
else
	$(error ERROR: Type of arch='$(ARCH) unsuported.)
endif

CFLAGS += -Wall -O3 -fopenmp
LDFLAGS = -lm -fopenmp

TARGET  = build/spmv
SRC_DIR = src
OBJ_DIR = build

$(sell mkdir -p $(OBJ_DIR))

SRCS := $(wildcard $(SRC_DIR)/*c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

spmv: driver_spmv.o $(OBJ_DIR)
	$(CC) $(LDFLAGS) $^ -o $@


clean:
	rm -rf build/*

