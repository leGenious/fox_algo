#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define BUFFSIZE 100
#define LEX(_i, _j, _dim) (_i*_dim[1]+_j) // had this idea way too late unfortunately

#define DEBUGPRINT(_fmt, ...) fprintf(stderr, "[file: %s, line: %d] " _fmt, __FILE__, __LINE__, __VA_ARGS__)


void read_dimensions(FILE* file, int* dim, int me)
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

void read_matrix(FILE* file, double* mat_local, int* dim, int* dim_local, int me, int*grid_dim, int q)
{
	char strbuffer[BUFFSIZE];
	MPI_Status status;
	if ( me == 0 )
	// read matrix A
	{
		double *buffer = (double*) malloc(sizeof(double)*dim_local[1]);
		double tmp;
		// please excuse me for the quadrouple-for loop, it's more readable this
		// way
		for ( int dest_row=0; dest_row<q; ++dest_row )
		{
			for ( int i=0; i<dim_local[0]; ++i )
			// loop through rows
			{
				int tag = i;
				for ( int dest_col=0; dest_col<q; ++dest_col )
				{
					for ( int j=0; j<dim_local[1]; ++j )
					{
						fgets(strbuffer, BUFFSIZE, file);
						sscanf(strbuffer, "%lf", &tmp);
						buffer[j] = tmp;
					}
					int dest = grid_dim[1]*dest_row+dest_col;
#ifdef DEBUG
					DEBUGPRINT("sending row %d, part %d to proc %d, content: %lf\n", i, dest_col, dest, buffer[0]);
#endif
					if ( dest != 0 )
					{
						MPI_Send(buffer, dim_local[1], MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
					}
					else
					{
						memcpy(&mat_local[(i)*dim_local[1]], buffer, sizeof(double)*dim_local[1]);
					}
				}
			}
		}
	}
	else
	// recieve local rows
	{
		for ( int i=0; i<dim_local[0]; ++i )
		{
#ifdef DEBUG
			DEBUGPRINT("proc %d recieving row %d from 0\n", me, i);
#endif
			MPI_Recv(&mat_local[i*dim_local[1]], dim_local[1], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
		}
	}
}

void calc_local_dimensions(int* dim_local, int*dim, int q)
{
	dim_local[0] = dim[0]/q;
	dim_local[1] = dim[1]/q;

	for (int i=0; i<2; ++i)
	{
		if ( dim_local[i]*q != dim[i] )
		{
			dim_local[i] += dim_local[i] % q;
		}
	}
}
