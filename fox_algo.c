#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define BUFFSIZE 100
#define LEX(_i, _j, _dim) (_i*_dim[1]+_j) // had this idea way too late unfortunately

#define DEBUGPRINT(_fmt, ...) fprintf(stderr, "[file: %s, line: %d] " _fmt, __FILE__, __LINE__, __VA_ARGS__)

int main(int argc, char** argv)
{
	FILE *matfile_A,
		 *matfile_B,
		 *matfile_C;

	MPI_Status status;

	char charbuffer[BUFFSIZE];

	double *A_local,
		   *B_local,
		   *C_local;

	int dim_A[2],
		dim_B[2],
		dim_C[2],
		dim_A_local[2],
		dim_B_local[2],
		dim_C_local[2];

	int me, np,
		grid_index[2],		// row and col rank of the individual procs
		grid_me;			// own rank on the grid
	int grid_dim[2],		// dimensions of the grid
		wrap_around[2],		// dimensions to be circular
		free_dim[2];		// for creation of row and column communicators

	MPI_Comm grid_comm, row_comm, col_comm;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	// make the grid
	int q = (int)sqrt(np);
	// TODO: add some error handeling if q is not a square number
	// Create the grid Comm and the row_comms
	grid_dim[0] = grid_dim[1] = q;
	wrap_around[0] = wrap_around[1] = 1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dim, wrap_around, 1, &grid_comm);
	MPI_Comm_rank(grid_comm, &grid_me);

	// create row_comm
	free_dim[0] = 0; free_dim[1] = 1;
	MPI_Cart_sub(grid_comm, free_dim, &row_comm);
	MPI_Comm_rank(row_comm, &grid_index[1]);

	// create col_comm
	free_dim[1] = 0; free_dim[0] = 1;
	MPI_Cart_sub(grid_comm, free_dim, &col_comm);
	MPI_Comm_rank(col_comm, &grid_index[0]);

#ifdef DEBUG
	fprintf(stderr, "proc %d has new ID %d and coords (%d,%d)\n", me, grid_me, grid_index[0], grid_index[1]);
#endif

	if ( me == 0 )
	// read in matrix dimensions
	{
		matfile_A = fopen(argv[1], "r");
		fgets(charbuffer, BUFFSIZE, matfile_A);
		sscanf(charbuffer, "%d", &dim_A[0]);
		fgets(charbuffer, BUFFSIZE, matfile_A);
		sscanf(charbuffer, "%d", &dim_A[1]);
	}
	MPI_Bcast(dim_A, 2, MPI_INT, 0, MPI_COMM_WORLD);

	// calculate local dimensions
	dim_A_local[0] = dim_A[0]/q;
	dim_A_local[1] = dim_A[1]/q;

	A_local = (double*) malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	if ( me == 0 )
	// read matrix A
	{
		double *buffer = (double*) malloc(sizeof(double)*dim_A_local[1]);
		double tmp;
		// please excuse me for the quadrouple-for loop, it's more readable this
		// way ( it really is, I've tried and I am sorry )
		for ( int dest_row=0; dest_row<q; ++dest_row )
		{
			for ( int i=0; i<dim_A_local[0]; ++i )
			// loop through rows
			{
				int tag = i;
				for ( int dest_col=0; dest_col<q; ++dest_col )
				{
					for ( int j=0; j<dim_A_local[1]; ++j )
					{
						fgets(charbuffer, BUFFSIZE, matfile_A);
						sscanf(charbuffer, "%lf", &tmp);
						buffer[j] = tmp;
					}
					int dest = grid_dim[1]*dest_row+dest_col;
#ifdef DEBUG
					DEBUGPRINT("sending row %d, part %d to proc %d, content: %lf\n", i, dest_col, dest, buffer[0]);
#endif
					if ( dest != 0 )
					{
						MPI_Send(buffer, dim_A_local[1], MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
					}
					else
					{
						memcpy(&A_local[(i)*dim_A_local[1]], buffer, sizeof(double)*dim_A_local[1]);
					}
				}
			}
		}
		// close file ONLY on proc 0
		fclose(matfile_A);
	}
	else
	// recieve local rows
	{
		for ( int i=0; i<dim_A_local[0]; ++i )
		{
#ifdef DEBUG
			DEBUGPRINT("proc %d recieving row %d from 0\n", me, i);
#endif
			MPI_Recv(&A_local[i*dim_A_local[1]], dim_A_local[1], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
		}
	}


#ifdef DEBUG
	for ( int i=0; i<dim_A_local[0]*dim_A_local[1]; ++i )
	{
		DEBUGPRINT("proc %d holds at pos %d: %lf\n", grid_me, i, A_local[i]);
	}
#endif

	// read matrix B
	if ( me == 0 )
	// read in matrix dimensions
	{
		matfile_B = fopen(argv[1], "r");
		fgets(charbuffer, BUFFSIZE, matfile_B);
		sscanf(charbuffer, "%d", &dim_B[0]);
		fgets(charbuffer, BUFFSIZE, matfile_B);
		sscanf(charbuffer, "%d", &dim_B[1]);
	}
	MPI_Bcast(dim_B, 2, MPI_INT, 0, MPI_COMM_WORLD);

	// calculate local dimensions
	dim_B_local[0] = dim_B[0]/q;
	dim_B_local[1] = dim_B[1]/q;

	B_local = (double*) malloc(sizeof(double)*dim_B_local[0]*dim_B_local[1]);

	if ( me == 0 )
	// read matrix B
	{
		double *buffer = (double*) malloc(sizeof(double)*dim_B_local[1]);
		double tmp;
		// please excuse me for the quadrouple-for loop, it's more readable this
		// way ( it really is, I've tried and I am sorry )
		for ( int dest_row=0; dest_row<q; ++dest_row )
		{
			for ( int i=0; i<dim_B_local[0]; ++i )
			// loop through rows
			{
				int tag = i;
				for ( int dest_col=0; dest_col<q; ++dest_col )
				{
					for ( int j=0; j<dim_B_local[1]; ++j )
					{
						fgets(charbuffer, BUFFSIZE, matfile_B);
						sscanf(charbuffer, "%lf", &tmp);
						buffer[j] = tmp;
					}
					int dest = grid_dim[1]*dest_row+dest_col;
#ifdef DEBUG
					DEBUGPRINT("sending row %d, part %d to proc %d, content: %lf\n", i, dest_col, dest, buffer[0]);
#endif
					if ( dest != 0 )
					{
						MPI_Send(buffer, dim_B_local[1], MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
					}
					else
					{
						memcpy(&B_local[(i)*dim_B_local[1]], buffer, sizeof(double)*dim_B_local[1]);
					}
				}
			}
		}
		// close file ONLY on proc 0 (otherwise procs > 0 will segfault at nil)
		fclose(matfile_B);
	}
	else
	// recieve local rows
	{
		for ( int i=0; i<dim_B_local[0]; ++i )
		{
#ifdef DEBUG
			DEBUGPRINT("proc %d recieving row %d from 0\n", me, i);
#endif
			MPI_Recv(&B_local[i*dim_B_local[1]], dim_B_local[1], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
		}
	}


#ifdef DEBUG
	for (int i=0; i<dim_A_local[0]*dim_A_local[1]; ++i)
	{
		DEBUGPRINT("proc %d A[%d]=%lf\n", me, i, A_local[i]);
	}
	for (int i=0; i<dim_B_local[0]*dim_B_local[1]; ++i)
	{
		DEBUGPRINT("proc %d B[%d]=%lf\n", me, i, B_local[i]);
	}
#endif

	// allocate c_local and initialize it to zeros
	C_local = (double*)calloc(sizeof(double),dim_A_local[0]*dim_B_local[1]);
	double* tmp = (double*)malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);
	dim_C_local[0] = dim_A_local[0];
	dim_C_local[1] = dim_B_local[1];

	// PERFORM THE MULTIPLICATION
	for (int stage=0; stage<q; ++stage)
	// LOOP THROUGH THE STAGES OF THE ALGO
	{
		// determine active sender of a_local
		int root = (grid_index[0]+stage)%q;
		// broadcast A_local through the row
		if ( grid_index[1] == root )
		{
			memcpy(tmp, A_local, sizeof(double)*dim_A_local[0]*dim_A_local[1]);
			//for (int i=0; i<dim_A_local[0]*dim_A_local[1]; ++i)
			//DEBUGPRINT("proc %d, stage %d sending A[%d]=%lf\n", me, stage, i, tmp[0]);
		}
		MPI_Bcast(tmp, dim_A_local[0]*dim_A_local[1], MPI_DOUBLE, root, row_comm);
		// do multiplication
		for (int i=0; i<dim_A_local[0]; ++i)
		// loop through rows of C
		{
			for (int j=0; j<dim_B_local[1]; ++j)
			// loop elements of C
			{
				for (int k=0; k<dim_B_local[0]; ++k)
				{
					// C(i,j) = sum_k A(i,k)*B(k,j);
					C_local[i*dim_B_local[1]+j] +=
						tmp[i*dim_A_local[1]+k]*B_local[k*dim_B_local[1]+j];
				}
			}
		}
		// do circular shift of B_local upwards
		int source = (grid_index[0]+1)%q;
		int dest = (grid_index[0]-1+q)%q;
//		DEBUGPRINT("proc %d (%d,%d), source: %d: dest: %d\n", me, grid_index[0], grid_index[1], source, dest);
		MPI_Sendrecv_replace(B_local, dim_B_local[0]*dim_B_local[1], MPI_DOUBLE, dest, stage, source, stage, col_comm, &status);
	}

#ifdef DEBUG
	for (int i=0; i<dim_A_local[0]*dim_B_local[1]; ++i)
	{
		DEBUGPRINT("proc %d has C[%d]=%lf\n", me, i, C_local[i]);
	}
#endif

	// PERFORM OUTPUT
	if ( me == 0 )
	// print matrix C dimensions
	{
		matfile_C = fopen(argv[3], "w");
		fprintf(matfile_C, "%d\n", dim_A[0]);
		fprintf(matfile_C, "%d\n", dim_B[1]);
	}

	if ( grid_index[1] == 0 )
		tmp = realloc(tmp, sizeof(double)*dim_B[1]); // one row of C
	else
		free(tmp);

	if ( grid_index[0] == 0 )
	{
		for (int i=0; i<dim_C_local[0]; ++i)
		// loop through the first block row
		{
			// Gather row i in proc 0
			MPI_Gather(&C_local[i*dim_C_local[1]], dim_C_local[1], MPI_DOUBLE, tmp, dim_C_local[1], MPI_DOUBLE, 0, row_comm);
			if ( me == 0 )
			{
				for (int j=0; j<dim_B[1]; ++j)
				{
					fprintf(matfile_C, "%lf\n", tmp[j]);
				}
			}
		}
	}

	for (int block_row=1; block_row<q; ++block_row)
	{
		if ( grid_index[0] == block_row )
		// gather individual rows
		{
			for (int i=0; i<dim_C_local[0]; ++i)
			{
				MPI_Gather(&C_local[i*dim_C_local[1]], dim_C_local[1], MPI_DOUBLE, tmp, dim_C_local[1], MPI_DOUBLE, 0, row_comm);
				if ( grid_index[1] == 0 )
				// send row to proc 0
				{
					int row_num = i+block_row*dim_C_local[0];
					DEBUGPRINT("proc %d, sending message %d\n", me, row_num);
					MPI_Send(tmp, dim_B[1], MPI_DOUBLE, 0, row_num, col_comm);
				}
			}
		}
		else if ( grid_index[1] == 0 )
		// recieve individual rows from proc block_row
		{
			for (int i=0; i<dim_C_local[0]; ++i)
			{
				DEBUGPRINT("proc %d, recieving message no %d from %d\n", me, i+block_row*dim_C_local[0], block_row);
				MPI_Recv(tmp, dim_B[1], MPI_DOUBLE, block_row, i+block_row*dim_C_local[0], col_comm, &status);
				for (int j=0; j<dim_B[1]; ++j)
				{
					fprintf(matfile_C, "%lf\n", tmp[j]);
				}
			}
		}
	}

	MPI_Finalize();
}
