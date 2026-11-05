#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv); // initialize MPI environment

    int rank; // process ID
    int size; // number of processes

    // get unique identifier of current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // communicator containing all processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from process %d of %d\n", rank, size);

    MPI_Finalize(); // shut down MPI environment

    return 0;
}