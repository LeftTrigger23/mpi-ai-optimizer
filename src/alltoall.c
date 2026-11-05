#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    // initialize MPI environment
    MPI_Init(&argc, &argv);

    // rank: unique ID of current process
    // size: total number of processes
    int rank, size;

    // get rank of current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // check whether user provided a message size
    if(argc != 2) {
        if(rank == 0) {
            printf("Usage: mpirun -np 4 ./alltoall <count>\n");
        }
        MPI_Finalize();
        return 1;
    }

    // convert command line argument into an integer
    int count = atoi(argv[1]);

    // ensure message size is positive
    if(count <= 0) {
        if(rank == 0) {
            printf("Error: count must be positive.\n");
        }

        MPI_Finalize();
        return 1;
    }

    // total number of integers sent and received by each process
    int total_elements = size * count;

    // allocate memory for sending array
    // each process need space for 'size' integers
    // it sends one integer to every process
    int *send_data = malloc(total_elements * sizeof(int));

    // allocate memory for receving array
    // each process recieves one integer from every process
    int *recv_data = malloc(total_elements * sizeof(int));

    // check whether memory allocation was successful
    if(send_data == NULL || recv_data == NULL) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // initialize sending arry with unique values
    // formula: rank * 10 + destination rank

    // 4 processes:
    // Rank 0: { 0,  1,  2,  3}
    // Rank 1: {10, 11, 12, 13}
    // Rank 2: {20, 21, 22, 23}
    // Rank 3: {30, 31, 32, 33}
    // Initialize the sending array for every destination
    for (int destination = 0; destination < size; destination++) {

        // Starting index of this destination's block
        int offset = destination * count;

        // Initialize all integers in the block
        for (int j = 0; j < count; j++) {

            // Generate a unique value based on the source rank,
            // destination rank, and position within the block
            send_data[offset + j] =
                rank * (size * count) + destination * count + j;
        }
    }

    // warmp-up iterations
    int warmup = 10;

    // perform communitcation without measuring execution time
    for(int i = 0; i < warmup; i++) {
        MPI_Alltoall(
            send_data, count, MPI_INT,
            recv_data, count, MPI_INT,
            MPI_COMM_WORLD
        );
    }

    // number of timed iterations
    int iterations = 100;

    // store total execution time
    double total_elapsed = 0.0;

    // synchronize all processes before starting measurement
    // ensure every process reaches the starting point
    // before any process begins timed communication
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0; i < iterations; i++) {
        // record starting time
        double start = MPI_Wtime();

        // execute communication operation
        MPI_Alltoall(
            send_data, count, MPI_INT,
            recv_data, count, MPI_INT,
            MPI_COMM_WORLD
        );

        // stop measuring
        double end = MPI_Wtime();

        // calculate elapsed time in seconds
        total_elapsed += end - start;
    }

    // calculate average execution time
    double average_elapsed = total_elapsed / iterations;

    // store max average time across all ranks
    double max_elapsed = 0.0;

    // find max elapsed time across all processes
    MPI_Reduce(&average_elapsed, &max_elapsed, 1,
    MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // display individual communication time
    /*
    printf("Rank %d: Communication time = %.9f seconds\n",
            rank, elapsed);
    
    // only rank 0 prints overall communication time
    if(rank == 0) {
        printf("Overall communication time = %.9f seconds\n", max_elapsed);
    }*/

    // print result of rank 0 only
    if(rank == 0) {
        printf("Iterations: %d\n", iterations);
        printf("Maximum average communication time = %.9f seconds\n", max_elapsed);
    }
    
    /*
    // print rank of current process
    printf("Rank %d receives: ", rank);

    // loop through every source process
    for(int source = 0; source < size; source++) {

        // find beginning of source's block
        int offset = source * count;
        printf("From rank %d ", source);

        // print every integer received from source
        for (int j = 0; j < count; j++) {
            printf("%d ", recv_data[offset + j]);
        }
        printf("\n");
    }*/

    // assume all received values are correctly initially
    int correct = 1;

    // check data received from every source process
    for(int source = 0; source < size; source++) {

        // find beginning of this source's received block
        int offset = source * count;

        // check every integer within block
        for(int j = 0; j < count; j++) {
            // caluclate expected value from source rank i
            int expected = 
            source * (size * count) + rank * count + j;

            // read actual value from received array
            int actual = recv_data[offset + j];

            // compare actual and expected
            if(actual != expected) {
                printf("Rank %d: ERROR from source %d at position %d "
                "(expected %d got %d)\n", rank, source, j, expected, actual);

                // mark verification failed
                correct = 0;
            }
        }
    }

    
    // print verification result for process
    if(correct) {
        printf("Rank %d: Verification PASSED\n", rank);
    }
    else {
        printf("Rank %d: Verification FAILED\n", rank);
    }

    // release dynamically allocated memory
    free(send_data);
    free(recv_data);

    // shut down MPI environment
    MPI_Finalize();

    return 0;
}