MPICC=mpicc
MPICCFLAGS=-lm -O3 -std=c99
MPIRUN=mpirun
TARGET=fox_algo
SRC=fox_algo.c

all: debug

debug: $(SRC)
	$(MPICC) $(MPICCFLAGS) -o $(TARGET) -g -DDEBUG $^

$(TARGET): $(SRC)
	$(MPICC) $(MPICCFLAGS) -o $@ $^
