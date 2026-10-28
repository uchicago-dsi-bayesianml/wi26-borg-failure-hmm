# Data

The trace extract that the notebooks read is not in this repository. The authors kept it in a Google Drive folder (`/content/drive/MyDrive/Bayesian Final Project/DATA`), and every notebook has a `path` or `DATA_DIR` cell near the top that points there. The two files that are here are the fitted model's parameters, which are enough for the decoding and validation steps and for the staff check.

## What is here

- `hmm_model_march11.pkl`: a dictionary with `pi` (3,), `A` (3, 3), `mu` (3, 5), `sigma` (3, 5), `features` (the five feature column names, in order), `K` = 3 and `norm_params` (below). Written at the end of `notebooks/hmm_model_v02.ipynb`; read by `notebooks/validation.ipynb` and by `staff/check_row_argmax.py`.
- `norm_params.pkl`: for each of eleven raw usage columns, the `eps` (1e-7) added before the log, and the mean and standard deviation of `log(x + eps)` over the training tasks. Both notebooks apply `(log(x + eps) - mean) / std` with these values, so the test tasks and the validation run are transformed with training-set statistics.

Both files are Python pickles of NumPy arrays and plain containers. `staff/check_row_argmax.py` shows how to load them with an unpickler that refuses anything but NumPy types.

## Where the data comes from

[Google ClusterData2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) (Tirmazi et al., "Borg: the Next Generation", EuroSys 2020): one month, May 2019, of traces from eight Borg cells of about 12,000 machines each. The team used cell *a*, bucket `gs://clusterdata_2019_a`, and three of its tables:

- `instance_usage`: per-task resource measurements in five-minute windows (average and maximum CPU and memory, assigned memory, page cache, cycles per instruction, and CPU usage percentile distributions). This cell's table is more than 12,000 compressed Parquet shards.
- `instance_events`: task lifecycle events with a type code. The ones used here are EVICT (4), FAIL (5) and KILL (7); the trace also has SUBMIT, QUEUE, ENABLE, SCHEDULE, FINISH, LOST and two UPDATE types.
- `collection_events`: job-level metadata, including priority, from which the team's priority bins and job outcomes were derived.

**Terms.** The trace and its documentation are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the trace and the paper if you use it.

## How the team built the extract

The steps are in `notebooks/pull_data.ipynb`; the loader class is also in `src/data_loader.py`. It reads shards straight from the bucket with DuckDB's `httpfs` extension, in parallel batches of shards.

1. A table of 1,200 jobs with a priority bin (2, 3 or 4) and an outcome (failed, finished, manual kill, other), `collections_per_priority_parquet.parquet`. It was built from `collection_events` outside this repository; the notebook only reads it.
2. Lifecycle events for those jobs from the first ten `instance_events` shards: 125,586 events for 41,063 tasks, 7,061 of which span a positive amount of time.
3. A stratified sample of up to 100 tasks per outcome and priority bin: 905 tasks.
4. Usage rows for those tasks, matched on the pair (`collection_id`, `instance_index`), saved as `instance_usage_training01_parquet.parquet`. The recorded run reads 500 of the usage shards and finds rows for 712 tasks, with a median of 46 rows per task. The extract that the modelling notebooks load has all 905 tasks and 3,436,845 rows, so a larger pull was made later and its run was not saved. To rebuild it, raise `shards` in the last `parallel_load` call; the full table is more than 12,000 shards, so expect the pull to take hours.
5. The events are saved as `instance_events_training01_parquet.parquet` and both files go to the Drive folder.

**A note on coverage.** The shards are not organized by task or by time, so a pull over part of them gives every task a random subset of its windows. In the recorded run the median task had about 94 minutes between consecutive rows, against the trace's five-minute measurement interval. The modelling notebooks treat consecutive rows as consecutive time steps regardless of the gap between them.

**Labels.** A task whose last lifecycle event is FAIL or KILL gets `last_event = 1` (140 of the 905 tasks); the split into 724 training and 181 test tasks is stratified on this flag. A *kill row* is the usage row with the latest `start_time` at or before each FAIL or KILL event of the task: 3,041 rows in the training tasks and 995 in the test tasks, out of 3.25 million after duplicate timestamps are dropped. KILL events are not separated into a user's kill, cleanup by a parent job, and a kill by the system.

**Crash loops.** The tasks with the most rows (97,593, 67,259, 66,760 and so on) all ended in FAIL or KILL and have rows seconds apart rather than five minutes: tasks that were restarted over and over, each restart leaving its own usage records. A handful of them account for a large share of all rows.
