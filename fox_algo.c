#include "fox_algo.h"

// TODO: make program work for matrices nxm where ( (n % q) != 0 ) || ( (m % q) != 0)

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

	int dim_local[3];
	int dim_A[2],
		dim_B[2],
		dim_C[2],
		dim_A_local[2],
		dim_B_local[2],
		dim_C_local[2];

	int me, np,
		grid_index[2];		// row and col rank of the individual procs
	int grid_dim[2],		// dimensions of the grid
		wrap_around[2],		// dimensions to be circular
		free_dim[2];		// for creation of row and column communicators

	MPI_Comm grid_comm, row_comm, col_comm;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
	// make the grid
	int q = (int)sqrt(np); // size of the qxq processor grid

	// crash (kind of) gracefully when q is not an int
	if ( floor(sqrtf( (double)np )) != sqrtf( (double)np ) )
	{
		MPI_Finalize();
		return -1;
	}

	// Create the grid Comm and the row_comms
	grid_dim[0] = grid_dim[1] = q;
	wrap_around[0] = wrap_around[1] = 1;
	MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dim, wrap_around, 1, &grid_comm);

	// create row_comm
	free_dim[0] = 0; free_dim[1] = 1;
	MPI_Cart_sub(grid_comm, free_dim, &row_comm);
	MPI_Comm_rank(row_comm, &grid_index[1]);

	// create col_comm
	free_dim[1] = 0; free_dim[0] = 1;
	MPI_Cart_sub(grid_comm, free_dim, &col_comm);
	MPI_Comm_rank(col_comm, &grid_index[0]);

	if ( me == 0 )
	// open the matrix files A & B
	{
		matfile_A = fopen(argv[1], "r");
	  	matfile_B = fopen(argv[2], "r");
	}

	read_dimensions(matfile_A, dim_A, me);
	read_dimensions(matfile_B, dim_B, me);

	calc_local_dimensions(dim_A_local, dim_A, q);
	calc_local_dimensions(dim_B_local, dim_B, q);

#ifdef DEBUG
	DEBUGPRINT("proc %d: local dim_A: %d,%d, local_dim_B: %d,%d\n", me, dim_A_local[0], dim_A_local[1], dim_B_local[0], dim_B_local[1]);
#endif

	// allocate and init local matrices to 0
	B_local = (double*) malloc(sizeof(double)*dim_B_local[0]*dim_B_local[1]);
	A_local = (double*) malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	memset(B_local, '\0', sizeof(double)*dim_B_local[0]*dim_B_local[1]);
	memset(A_local, '\0', sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	read_matrix(matfile_A, A_local, dim_A, dim_A_local, me, q);
	read_matrix(matfile_B, B_local, dim_B, dim_B_local, me, q);


#ifdef DEBUG
	DEBUGPRINT("proc %d: successfully read matrices\n", me);
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
	C_local = (double*)malloc(sizeof(double)*dim_A_local[0]*dim_B_local[1]);
	memset(C_local, '\0', dim_A_local[0]*dim_B_local[1]*sizeof(double));
	double* tmp = (double*)malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	dim_local[0] = dim_A_local[0];
	dim_local[1] = dim_A_local[1];
	dim_local[2] = dim_B_local[1];

	if ( me == 0 )
	{
		fclose(matfile_B);
		fclose(matfile_A);
	}

#ifndef TIMEIT
	fox_matmulmat(C_local,
			A_local,
			B_local,
			dim_local,
			q,
			grid_index,
			row_comm,
			col_comm);
#endif
#ifdef TIMEIT
	double time = fox_matmulmat_timed(C_local,
			A_local,
			B_local,
			dim_local,
			q,
			grid_index,
			row_comm,
			col_comm);
	if ( access( "timings.log", F_OK) != -1)
	{
		FILE* timings = fopen("timings.log", "a");
		fprintf(timings, "");
	}
	else
	{
		FILE* timings = fopen("timings.log", "w");
		fprintf(timings, "nprocs,m,k,n,calc\n");
		fprintf(timings, "%d,%d,%d,%d,%lf\n", np, dim_A[0], dim_A[1], dim_B[1], time);
	}
#endif

	DEBUGPRINT("proc %d successfully completed the multiplication\n", me);
#ifdef DEBUG
	for (int i=0; i<dim_A_local[0]*dim_B_local[1]; ++i)
	{
		DEBUGPRINT("proc %d has C[%d]=%lf\n", me, i, C_local[i]);
	}
#endif

	if ( me == 0 )
	{
		matfile_C = fopen(argv[3], "w");
	}

	dim_C[0] = dim_A[0];
	dim_C[1] = dim_B[1];

	dim_C_local[0] = dim_A_local[0];
	dim_C_local[1] = dim_B_local[1];

	write_matrix(matfile_C, C_local, dim_C, dim_C_local, grid_index, me, q, row_comm, col_comm);

	if ( me == 0 )
	{
		fclose(matfile_C);
	}

	MPI_Finalize();
}
