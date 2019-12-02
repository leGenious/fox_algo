MPICC = mpicc
MPICCFLAGS = -lm -O3 -std=c99
LFLAGS = -lopenblas -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib
MPIRUN = mpirun
TARGET = fox_algo
SRC = fox_algo.c
DEP = fox_algo.dep

MAKEDEP = mpicc -MM

LD_LIBRARY_PATH = /opt/OpenBLAS/lib

all: debug

debug: $(SRC)
	$(MPICC) $(MPICCFLAGS) $(LFLAGS) -o $(TARGET) -g -DDEBUG $^

%.dep: %.c
	$(MAKEDEP) $< | sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@

$(TARGET): $(SRC)
	$(MPICC) $(MPICCFLAGS) $(LFLAGS) -o $@ $^

-include $(DEP)
