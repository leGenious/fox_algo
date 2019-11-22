MPICC=/cluster/mpi/openmpi/1.6.5-gcc4.4.7/bin/mpicc
MPICCFLAGS=-lm -O3 -std=c99
MPIRUN=mpirun
TARGET=fox_algo
SRC=fox_algo.c

debug: $(SRC)
	$(MPICC) $(MPICCFLAGS) -o $(TARGET) -g -DDEBUG $^

$(TARGET): $(SRC)
	$(MPICC) $(MPICCFLAGS) -o $@ $^
