#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    int send_data, recv_data;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != 2) {
        if(rank == 0) {
            printf("Please run with exactly 2 processes.\n");
        }

        MPI_Finalize();
        return 1;
    }

    // exchange data with each other here
    send_data = (rank + 1) * 10;
    int other_rank = 1 - rank;
    
    // coordinated send and receive
    MPI_Sendrecv(
        &send_data, 1, MPI_INT, other_rank, 0,
        &recv_data, 1 , MPI_INT, other_rank, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    printf("Rank %d send %d and received %d\n", 
        rank, send_data, recv_data);
    
    MPI_Finalize();

    return 0;
}