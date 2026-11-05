# AI-based MPI communication optimizer

This Apple Silicon prototype compares two implementations of fixed-count all-to-all exchange and tests whether a small decision tree can choose between them. The archived local measurements favor the library collective. The launcher defaults to that collective and exposes the fitted tree for experiments.

## Setup and quick start

From the project directory, use an MPI installation that provides `mpicc` and `mpirun` and the existing Python virtual environment with pandas and scikit-learn. Verification used arm64 macOS, Open MPI 5.0.11, Python 3.15.0, pandas 3.0.6, and scikit-learn 1.9.1. Those versions describe the verification environment. The archived timings may come from a different setup.

```sh
cd ~/mpi-ai-optimizer
.venv/bin/python train_selector.py
.venv/bin/python run_selector.py 2 1
.venv/bin/python run_selector.py 2 1 --experimental-tree
```

If starting from source in a fresh copy, create the environment with `python3 -m venv .venv` and install dependencies with `.venv/bin/python -m pip install pandas scikit-learn`. Build the executables alongside the scripts in that copy.

```sh
mpicc -O2 alltoall.c -o alltoall
mpicc -O2 pairwise_alltoall.c -o pairwise_alltoall
mpirun -np 4 ./alltoall 100
mpirun -np 4 ./pairwise_alltoall 100
```

Add `--local-unbound` to the selector on local Open MPI setups where automatic placement fails. It requests explicit localhost slots, maps ranks by slot, and disables core binding. Record placement when comparing timings.

## Interactive MPI explorer

From the repository root, open the explorer in your default browser:

```sh
.venv/bin/python visualize_mpi.py
```

- **Compare timings.** Change the number of ranks and integers per destination to see the archived median for each algorithm. Configurations outside the benchmark grid have no archived timings.
- **Preview the selector.** Toggle the experimental tree to see which executable the launcher would choose. The explorer fits the same tree using the current training CSV.
- **Trace the exchange.** With 4 ranks, choose rank 0 and round 1. Rank 0 sends to rank 1 and receives from rank 3. Change the round to see how the custom `MPI_Sendrecv` schedule rotates through peers. The MPI library chooses the internal schedule for `MPI_Alltoall`.
- **Run MPI locally.** Copy the displayed command into Terminal to launch the selected executable and verify the received data. The explorer itself displays archived measurements and does not launch MPI.

To generate an HTML snapshot without opening the browser, run `.venv/bin/python visualize_mpi.py --output mpi-explorer-snapshot.html`. The destination must be a new file.

## Exchange and correctness

Each of the P ranks sends a block of count integers to every rank, including itself. Its send buffer is indexed by destination and its receive buffer by source. Both contain P × count integers per rank. For source s, destination d, and offset j, the sent value is s × P × count + d × count + j.

- `alltoall.c` delegates the exchange to `MPI_Alltoall`. The installed MPI library can select its own collective algorithm. [MPI_Alltoall reference](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Alltoall.3.html)
- `pairwise_alltoall.c` performs P rounds of blocking `MPI_Sendrecv`. In round k, rank r sends to (r + k) mod P and receives from (r − k + P) mod P. The sender and receiver agree in each round, and k = 0 handles self-exchange. [MPI_Sendrecv reference](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Sendrecv.3.html)

After timing, each rank compares every received element with the expected value for its source, destination and offset. The original programs print per-rank verification messages. The benchmark driver and selector inspect those messages because the C programs can return zero after reporting a data mismatch. The selector requires the expected pass count and rejects a reported failure or nonzero launcher exit.

## Benchmarking and datasets

Both executables run 10 warmups, synchronize with a barrier before timing, and use `MPI_Wtime` around each of 100 exchanges. Each rank averages its times and rank zero reports the maximum of the rank averages using `MPI_Reduce` with `MPI_MAX`. This measures the slowest rank's average exchange time. It excludes MPI startup, allocation, warmups, the barrier, verification, and selector overhead.

`benchmark.py` tests 2, 4 and 8 processes crossed with counts 1, 10, 100 and 1000 integers per destination. It launches both programs five times per configuration, always standard first, and saves each program's median of five run-level measurements in `repeated_results.csv` in seconds. Raw repetitions are absent from that archive, so it cannot establish run-to-run spread, stability of the winner, or confidence intervals.

`prepare_data.py` creates one row per configuration in `training_data.csv`. Label 0 means standard was faster or tied and label 1 means pairwise was faster. The sole label 1 is (2, 1), where medians were 5.740 µs standard and 5.620 µs pairwise. The 0.120 µs difference is noise-prone. Earlier CSVs `automated_results.csv` and `benchmark_results.csv` are retained as project history.

Run `benchmark.py` followed by `prepare_data.py` only in a separate working copy to preserve the originals. Both scripts write fixed filenames in the current directory and overwrite those files. Copy or build the MPI executables into that working copy first because the benchmark uses relative executable paths.

## Step 13 · Decision tree and evaluation

`train_selector.py` constructs X from `processes` and `count`, and y from `label`. The model is a scikit-learn classifier with maximum depth two and a fixed random seed. No timing columns are used as features. The script rejects duplicate configuration keys so a held-out row cannot have an identical configuration in training. Timings are joined after prediction to compute regret, the chosen median minus the lower of the two medians for that configuration. [DecisionTreeClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

Leave-one-configuration-out evaluation fits a new tree on 11 rows to predict the twelfth. The always-standard baseline predicts label 0 for every row. Confusion matrices below use actual labels as rows and predicted labels as columns, both ordered 0 then 1.

| Policy | Correct | Confusion matrix | Mean regret per configuration |
| --- | ---: | --- | ---: |
| Always standard | 11/12 | [[11, 0], [1, 0]] | 0.010 µs |
| Held-out tree | 9/12 | [[9, 2], [1, 0]] | 0.097 µs |

When the only pairwise-winning configuration is held out, the training fold contains only standard labels. The tree misses that configuration and selects pairwise incorrectly at (2, 10) and (4, 1). These results are descriptive of this small grid. There is no independent test campaign and one positive example cannot support a stable estimate of pairwise recall.

The tree fitted on all 12 rows predicts pairwise when count ≤ 5.5 and processes ≤ 3. It predicts standard otherwise. That fits all 12 archived labels, including the one weak pairwise label. The script prints the rules and both fitted and held-out predictions for each row.

## Step 14 · Selector and end-to-end verification

`run_selector.py PROCESSES COUNT` selects one executable before starting MPI, so every rank runs the same implementation. The default is standard. With `--experimental-tree`, it fits the same tree on the archived labels and uses it only on exact configurations present in the training grid. Outside the grid it falls back to standard. The Python launcher runs once per job and its startup and fitting overhead are outside the C timing.

The following cases were run using `--local-unbound`. Every requested rank passed buffer verification.

| Processes | Count | Mode | Executable selected |
| ---: | ---: | --- | --- |
| 2 | 1 | Default | `alltoall` |
| 2 | 1 | Experimental tree | `pairwise_alltoall` |
| 4 | 100 | Experimental tree | `alltoall` |
| 2 | 2 | Experimental tree, outside grid | `alltoall` |
| 8 | 1000 | Default | `alltoall` |

A zero process count was rejected before launch. Original top-level files matched their pre-change SHA-256 hashes. The smoke runs validate dispatch and data exchange. Performance conclusions use the archived benchmark and remain limited to its conditions.

## Limitations and future HPC work

- Expand the grid with more counts around observed transitions, larger messages, and more process counts. Record message bytes as well as integer counts, because transfer costs depend on bytes and the MPI datatype.
- Record raw run-level measurements, randomize or counterbalance execution order, and repeat across independent sessions. Store build flags, MPI version and collective settings, machine load, rank placement, binding, and node allocation. Report spread and uncertainty on paired differences, and define a practical speedup threshold before analyzing results. Keep standard as the default when evidence is ambiguous.
- Use a fresh benchmark campaign to test any tuned policy. With more observations, keep repeated runs of the same configuration together in evaluation splits and hold out whole sessions, rank counts, or allocations to match the intended deployment. Compare performance regret and false pairwise selections alongside accuracy.
- On a CARC or other HPC system, collect separate single-node and multi-node runs under the scheduler. Record node count, ranks per node, CPU binding, interconnect, and MPI configuration. A model trained only on process count and message count cannot distinguish those placements. Compare against the cluster's default collective and any explicitly configured library algorithms.
- If selection later moves inside an MPI application, make the choice consistent across ranks, for example by broadcasting one decision before the exchange. Measure selection overhead, document communicator and datatype support, and benchmark within a representative application.
