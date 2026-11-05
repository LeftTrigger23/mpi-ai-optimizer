import subprocess
import csv
import statistics


# Run an MPI benchmark and return its communication time
def run_benchmark(program, processes, count):

    # Construct the MPI command
    command = [
        "mpirun",
        "-np",
        str(processes),
        program,
        str(count)
    ]

    # Execute the MPI program and capture its output
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    # Check that every MPI process passed verification
    passed = result.stdout.count("Verification PASSED")

    if passed != processes:
        raise RuntimeError(
            f"Verification failed for {program}: "
            f"expected {processes} passes, found {passed}"
        )

    # Search for the communication time
    for line in result.stdout.splitlines():

        if "Maximum average communication time =" in line:

            # Extract the number after the equals sign
            time_text = line.split("=")[1].strip().split()[0]

            # Convert the extracted text into a number
            communication_time = float(time_text)

            # Return the communication time
            return communication_time

    # Raise an error if no communication time was found
    raise RuntimeError(f"No communication time found for {program}")


# Different numbers of MPI processes
process_counts = [2, 4, 8]

# Different numbers of integers per destination
message_counts = [1, 10, 100, 1000]

# Number of independent benchmark repetitions
repetitions = 5


# Open a CSV file to save the benchmark results
with open("repeated_results.csv", "w", newline="") as file:

    # Create a CSV writer
    writer = csv.writer(file)

    # Write the column headers
    writer.writerow([
        "processes",
        "count",
        "standard_time",
        "pairwise_time"
    ])

    # Test each process count
    for processes in process_counts:

        # Test each message size
        for count in message_counts:

            # Store measurements from repeated independent runs
            standard_times = []
            pairwise_times = []

            # Repeat each benchmark configuration
            for repetition in range(repetitions):

                # Run standard MPI_Alltoall
                standard_time = run_benchmark(
                    "./alltoall",
                    processes,
                    count
                )

                # Run custom pairwise all-to-all
                pairwise_time = run_benchmark(
                    "./pairwise_alltoall",
                    processes,
                    count
                )

                # Store the execution times
                standard_times.append(standard_time)
                pairwise_times.append(pairwise_time)

            # Calculate median communication times
            standard_time = statistics.median(standard_times)
            pairwise_time = statistics.median(pairwise_times)

            # Save the median results to CSV
            writer.writerow([
                processes,
                count,
                f"{standard_time:.9f}",
                f"{pairwise_time:.9f}"
            ])

            # Display progress
            print(f"Completed: processes={processes}, count={count}")
            print(f"  Standard median: {standard_time:.9f} seconds")
            print(f"  Pairwise median: {pairwise_time:.9f} seconds")
            print()


print("Benchmarking complete!")
print("Results saved to repeated_results.csv")