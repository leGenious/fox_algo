#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define BUFFSIZE 100
#define LEX(_i, _j, _dim) (_i*_dim[1]+_j) // had this idea way too late unfortunately

#define DEBUGPRINT(_fmt, ...) fprintf(stderr, "[file: %s, line: %d] " _fmt, __FILE__, __LINE__, __VA_ARGS__)


void read_dimensions(FILE* file, int* dim, int me, int np)
{
	char buffer[BUFFSIZE];
	if ( me == 0 )
	{
		fgets(buffer, BUFFSIZE, file);
		sscanf(buffer, "%d", &dim[0]);
		fgets(buffer, BUFFSIZE, file);
		sscanf(buffer, "%d", &dim[1]);
	}
	MPI_Bcast(dim, 2, MPI_INT, 0, MPI_COMM_WORLD);
}
