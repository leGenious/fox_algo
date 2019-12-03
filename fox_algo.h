#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include <cblas.h>

#define BUFFSIZE 100
#define LEX(_i, _j, _dim) (_i*_dim[1]+_j) // had this idea way too late unfortunately

#define DEBUGPRINT(_fmt, ...) fprintf(stderr, "[file: %s, line: %d] " _fmt, __FILE__, __LINE__, __VA_ARGS__)

void write_matrix(FILE* file, double* mat_local, int* dim, int* dim_local, int* grid_index, int me, int q, MPI_Comm row_comm, MPI_Comm col_comm);
void read_dimensions(FILE* file, int* dim, int me);
void fox_matmulmat(double* C_local, double* A_local, double* B_local, int* dim_local, int q, int* grid_index, MPI_Comm row_comm, MPI_Comm col_comm);
void calc_local_dimensions(int* dim_local, int*dim, int q);
double fox_matmulmat_timed(double* C_local, double* A_local, double* B_local, int* dim_local, int q, int* grid_index, MPI_Comm row_comm, MPI_Comm col_comm);
void read_matrix(FILE* file, double* mat_local, int* dim, int* dim_local, int me, int q);


void read_dimensions(FILE* file,	// [in] file to be read
		int* dim,					// [out] dimensions
		int me)						// [in] index of the processor
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

void fox_matmulmat(double* C_local, // [out] output local matrix
		double* A_local,			// [in] first multiplicant
		double* B_local,			// [in] first multiplicant
		int* dim_local,				// [in] dimensions of the local matrix [3]
		int q,						// [in] dimension of the (sqare) processor grid
		int *grid_index,			// [in] index of the processor on the grid [2]
		MPI_Comm row_comm,			// [in] a row communicator on the grid
		MPI_Comm col_comm)			// [in] a col communicator on the grid
{
	MPI_Status status;
	double* tmp = (double*)malloc(sizeof(double)*dim_local[0]*dim_local[1]);
	// PERFORM THE FOX ALGORITHM
	for (int stage=0; stage<q; ++stage)
	// loop through the algo stages
	{
		// determine active sender of a_local
		int root = (grid_index[0]+stage)%q;
		// broadcast A_local through the row
		if ( grid_index[1] == root )
			MPI_Bcast(A_local, dim_local[0]*dim_local[1], MPI_DOUBLE, root, row_comm);
		else
			MPI_Bcast(tmp, dim_local[0]*dim_local[1], MPI_DOUBLE, root, row_comm);
		// do multiplication
		if ( grid_index[1] != root )
		{
		cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasNoTrans,
				dim_local[0],
				dim_local[2],
				dim_local[1],
				1,
				tmp,
				dim_local[1],
				B_local,
				dim_local[2],
				1,
				C_local,
				dim_local[2]);
		}
		else
		{
		cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasNoTrans,
				dim_local[0],
				dim_local[2],
				dim_local[1],
				1,
				A_local,
				dim_local[1],
				B_local,
				dim_local[2],
				1,
				C_local,
				dim_local[2]);
		}
		// do circular shift of B_local upwards
		int source = (grid_index[0]+1)%q;
		int dest = (grid_index[0]-1+q)%q;
		MPI_Sendrecv_replace(B_local, dim_local[1]*dim_local[2], MPI_DOUBLE, dest, stage, source, stage, col_comm, &status);
	}
}

double fox_matmulmat_timed(
		double* C_local,			// [out] output local matrix
		double* A_local,            // [in] first multiplicant
		double* B_local,            // [in] first multiplicant
		int* dim_local,             // [in] dimensions of the local matrix [2]
		int q,                      // [in] dimension of the (sqare) processor grid
		int* grid_index,            // [in] index of the processor on the grid [2]
		MPI_Comm row_comm,          // [in] a row communicator on the grid
		MPI_Comm col_comm)          // [in] a col communicator on the grid
	// return the walltime of the implementation
{
	MPI_Barrier(MPI_COMM_WORLD);
	double time = MPI_Wtime();
	fox_matmulmat(C_local,
			A_local,
			B_local,
			dim_local,
			q,
			grid_index,
			row_comm,
			col_comm);
	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;
	return time;
}

void read_matrix(
		FILE* file,					// [in] file to be read
		double* mat_local,			// [out] local matrix part to be filled
		int* dim,					// [in] dimensions of the matrix [2]
		int* dim_local,				// [in] dimensions of the local part [2]
		int me,						// [in] my rank
		int q)						// [in] size of the processor grid
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
					int dest = q*dest_row+dest_col;
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

void calc_local_dimensions(
		int* dim_local,				// [out] local matrix part dimension
		int*dim,					// [in] global matrix dimension
		int q)						// [in] size of the processor grid
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

void write_matrix(
		FILE* file,					// [in] file to be written to
		double* mat_local,			// [in] local matrix to be written
		int* dim,					// [in] dimension of the matrix [2]
		int* dim_local,				// [in] dimension of the local parts
		int* grid_index,			// [in] index of the processors on the grid
		int me,						// [in] global proecssor rank
		int q,						// [in] size of the processor grid
		MPI_Comm row_comm,			// [in] a communicator through the rows of the grid
		MPI_Comm col_comm)			// [in] a communicator through the cols of the grid
{
	MPI_Status status;
	double * tmp;
	// PERFORM OUTPUT
	if ( me == 0 )
	// print matrix C dimensions
	{
		fprintf(file, "%d\n", dim[0]);
		fprintf(file, "%d\n", dim[1]);
		tmp = malloc(sizeof(double)*dim[1]);
	}
	else if ( grid_index[1] == 0 )
	{
		tmp = malloc(sizeof(double)*dim[1]);
	}

	if ( grid_index[0] == 0 )
	{
		for (int i=0; i<dim_local[0]; ++i)
		// loop through the first block row
		{
			// Gather row i in proc 0
			MPI_Gather(&mat_local[i*dim_local[1]], dim_local[1], MPI_DOUBLE, tmp, dim_local[1], MPI_DOUBLE, 0, row_comm);
			if ( me == 0 )
			{
				for (int j=0; j<dim[1]; ++j)
				{
					fprintf(file, "%lf\n", tmp[j]);
				}
			}
		}
	}


	for (int block_row=1; block_row<q; ++block_row)
	{
		if ( grid_index[0] == block_row )
		// gather individual rows
		{
			for (int i=0; i<dim_local[0]; ++i)
			{
				MPI_Gather(&mat_local[i*dim_local[1]], dim_local[1], MPI_DOUBLE, tmp, dim_local[1], MPI_DOUBLE, 0, row_comm);
				if ( grid_index[1] == 0 )
				// send row to proc 0
				{
					int row_num = i+block_row*dim_local[0];
					MPI_Send(tmp, dim[1], MPI_DOUBLE, 0, row_num, col_comm);
				}
			}
		}
		else if ( grid_index[1] == 0 )
		// recieve individual rows from proc block_row
		{
			for (int i=0; i<dim_local[0]; ++i)
			{
				MPI_Recv(tmp, dim[1], MPI_DOUBLE, block_row, i+block_row*dim_local[0], col_comm, &status);
				for (int j=0; j<dim[1]; ++j)
				{
					fprintf(file, "%lf\n", tmp[j]);
				}
			}
		}
	}
}
