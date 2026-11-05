import csv


# Read the benchmark results
with open("repeated_results.csv", "r", newline="") as file:

    reader = csv.DictReader(file)

    # Store labeled benchmark results
    labeled_data = []

    for row in reader:

        # Extract the configuration
        processes = int(row["processes"])
        count = int(row["count"])

        # Extract the measured communication times
        standard_time = float(row["standard_time"])
        pairwise_time = float(row["pairwise_time"])

        # Determine which algorithm is faster
        if standard_time <= pairwise_time:
            label = 0
        else:
            label = 1

        # Store the configuration and its label
        labeled_data.append([
            processes,
            count,
            label
        ])


# Save the labeled dataset
with open("training_data.csv", "w", newline="") as file:

    writer = csv.writer(file)

    # Write the header
    writer.writerow([
        "processes",
        "count",
        "label"
    ])

    # Write the labeled results
    writer.writerows(labeled_data)


print("Dataset preparation complete!")
print("Results saved to training_data.csv")