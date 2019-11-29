#include "fox_algo.h"

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
		dim_B_local[2];

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
	if ( floor(sqrtf( (double)np )) != q )
		return -1;

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

	// allocate and init local matrices to 0
	B_local = (double*) malloc(sizeof(double)*dim_B_local[0]*dim_B_local[1]);
	A_local = (double*) malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	memset(B_local, '\0', sizeof(double)*dim_B_local[0]*dim_B_local[1]);
	memset(A_local, '\0', sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	read_matrix(matfile_A, A_local, dim_A, dim_A_local, me, grid_dim, q);
	read_matrix(matfile_B, B_local, dim_B, dim_B_local, me, grid_dim, q);

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
	C_local = (double*)malloc(sizeof(double)*dim_A_local[0]*dim_B_local[1]);
	memset(C_local, '\0', dim_A_local[0]*dim_B_local[1]*sizeof(double)); // apparently faster than calloc
	double* tmp = (double*)malloc(sizeof(double)*dim_A_local[0]*dim_A_local[1]);

	// PERFORM THE FOX ALGORITHM
	for (int stage=0; stage<q; ++stage)
	// loop through the algo stages
	{
		// determine active sender of a_local
		int root = (grid_index[0]+stage)%q;
		// broadcast A_local through the row
		if ( grid_index[1] == root )
		{
			memcpy(tmp, A_local, sizeof(double)*dim_A_local[0]*dim_A_local[1]);
			//for (int i=0; i<dim_A_local[0]*dim_A_local[1]; ++i)
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
		DEBUGPRINT("B is %dx%d\n", dim_B[0], dim_B[1]);
		DEBUGPRINT("A is %dx%d\n", dim_A[0], dim_A[1]);
	}

	if ( grid_index[1] == 0 )
		tmp = realloc(tmp, sizeof(double)*dim_B[1]); // one row of C
	else
		free(tmp);

	if ( grid_index[0] == 0 )
	{
		for (int i=0; i<dim_A_local[0]; ++i)
		// loop through the first block row
		{
			// Gather row i in proc 0
			MPI_Gather(&C_local[i*dim_B_local[1]], dim_B_local[1], MPI_DOUBLE, tmp, dim_B_local[1], MPI_DOUBLE, 0, row_comm);
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
			for (int i=0; i<dim_A_local[0]; ++i)
			{
				MPI_Gather(&C_local[i*dim_B_local[1]], dim_B_local[1], MPI_DOUBLE, tmp, dim_B_local[1], MPI_DOUBLE, 0, row_comm);
				if ( grid_index[1] == 0 )
				// send row to proc 0
				{
					int row_num = i+block_row*dim_A_local[0];
					MPI_Send(tmp, dim_B[1], MPI_DOUBLE, 0, row_num, col_comm);
				}
			}
		}
		else if ( grid_index[1] == 0 )
		// recieve individual rows from proc block_row
		{
			for (int i=0; i<dim_A_local[0]; ++i)
			{
				MPI_Recv(tmp, dim_B[1], MPI_DOUBLE, block_row, i+block_row*dim_A_local[0], col_comm, &status);
				for (int j=0; j<dim_B[1]; ++j)
				{
					fprintf(matfile_C, "%lf\n", tmp[j]);
				}
			}
		}
	}

	MPI_Finalize();
}
