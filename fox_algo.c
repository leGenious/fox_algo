#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define BUFFSIZE 100

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
		for (int dest_row=0; dest_row<q; ++dest_row)
		{
			for (int i=0; i<dim_A_local[0]; ++i)
			// loop through rows
			{
				int tag = i;
				for (int dest_col=0; dest_col<q; ++dest_col)
				{
					for (int j=0; j<dim_A_local[1]; ++j)
					{
						fgets(charbuffer, BUFFSIZE, matfile_A);
						sscanf(charbuffer, "%lf", &tmp);
						buffer[j] = tmp;
					}
					int dest = grid_dim[1]*dest_row+dest_col;
					if ( dest != 0 )
					{
						MPI_Send(buffer, dim_A_local[1], MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
					}
					else
					{
						memcpy(&A_local[dest_row*dim_A_local[1]], buffer, dim_A_local[1]);
					}
				}
			}
		}
	}
	else
	// recieve local rows
	{
		for (int i=0; i<dim_A_local[0]; ++i)
		{
			MPI_Recv(&A_local[i*dim_A_local[1]], dim_A_local[1], MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
		}
	}

	for (int i=0; i<dim_A_local[0]*dim_A_local[1]; ++i)
	{
		fprintf(stderr, "proc %d holds at pos %d: %lf\n", grid_me, i, A_local[i]);
	}

	// read in the matrix
	MPI_Finalize();
}
