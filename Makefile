MPICC = mpicc
MPICCFLAGS = -lm -O3 -std=c99 -DNOBLAS
LFLAGS = -lopenblas -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib
MPIRUN = mpirun
MAKEDEP = mpicc -MM
TARGET = fox_algo
SRC = fox_algo.c
DEP = fox_algo.dep


LD_LIBRARY_PATH = /opt/OpenBLAS/lib

time: $(SRC)
	$(MPICC) $(MPICCFLAGS) $(LFLAGS) -o $(TARGET) -DTIMEIT $^

all: $(TARGET)

debug: $(SRC)
	$(MPICC) $(MPICCFLAGS) $(LFLAGS) -o $(TARGET) -g -DDEBUG $^

%.dep: %.c
	$(MAKEDEP) $< | sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@

$(TARGET): $(SRC)
	$(MPICC) $(MPICCFLAGS) $(LFLAGS) -o $@ $^

-include $(DEP)
