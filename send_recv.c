#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    MPI_Init (&argc, &argv); // initialize

    int rank;
    int data;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // determine rank

    if(rank == 0) {
        // 0 sends integer 42 to rank 1
        data = 42;
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0 sent: %d\n", data);
    }
    else if(rank == 1) {
        // waits for matching message from rank 0 and stores integer
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received %d\n", data);
    }

    MPI_Finalize();

    return 0;
}