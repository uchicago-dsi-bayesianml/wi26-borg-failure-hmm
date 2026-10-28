<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian Job Failure Prediction in Distributed Systems

> **Example project from Winter 2026.** This is a copy of [djdhawal/bayesian-failure-prediction-on-cluster-traces](https://github.com/djdhawal/bayesian-failure-prediction-on-cluster-traces), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project). The trace data is public under CC BY 4.0 but is not in this repository; the authors' extract lived in a Google Drive folder, and [data/README.md](data/README.md) explains how it was built.

Tasks on a large cluster rarely die out of nowhere. A task that is about to be killed often spends a while looking different from a task that is working. Can a hidden Markov model recover such states from resource telemetry alone, without ever seeing a failure label, and do the recovered states line up with the failures? The team took one month of Google's 2019 Borg traces, sampled 905 tasks across priority tiers and job outcomes, and fit a three-state Gaussian HMM to five log-transformed usage features by stochastic variational inference in NumPyro, with the hidden states enumerated exactly inside the objective. They then decoded every row with a hand-written forward filter and Viterbi and asked which state tasks were in when they were killed. One state, the one in which the CPU counters read close to zero, holds about four in five kill rows, which the write-up reads as a degraded, pre-failure regime. One teammate also wrote the forward algorithm into the model so that the hidden states are summed out and sampled the parameters with NUTS, and another trained a transformer to forecast decoded states.

## Summary

**Data.** Google's ClusterData2019 trace for cell *a*: per-task resource usage in five-minute windows, task lifecycle events and job metadata, read straight from the public Cloud Storage bucket with DuckDB. From 1,200 jobs in three priority bins the team sampled up to 100 tasks per combination of priority bin and job outcome, 905 tasks, and pulled their usage rows: 3.4 million rows, 3.25 million after the modelling notebook drops duplicate timestamps. A task whose last lifecycle event is FAIL or KILL is labelled as such (140 tasks against 765). Tasks are split 80/20 with that label stratified: 724 training tasks and 181 test tasks.

**Features.** Average CPU, the 90th percentile of CPU within the window, average memory, assigned memory, and a burstiness measure (the 90th minus the 10th CPU percentile), each log-transformed after adding 10⁻⁷ and standardized with training-set statistics. Missing values become zero before the log.

**Model.** A Gaussian HMM with three states: Dirichlet(1) priors on the initial distribution and on each row of the transition matrix, Normal(0, 1) priors on the per-state means and Exponential(1) priors on the per-state scales, with the five features independent given the state. The hidden states are enumerated exactly by `TraceEnum_ELBO`, the continuous parameters get a mean-field Gaussian guide, and Adam with gradient clipping and a decaying learning rate runs for 1,000 steps on mini-batches of four tasks, each cut to its first ten windows. Two hundred draws from the fitted guide are averaged into point estimates of the initial distribution, the transition matrix, the means and the scales, which are saved in `data/hmm_model_march11.pkl`.

**Decoding and validation.** With those point estimates, a NumPy forward filter gives filtered state probabilities for every row and Viterbi gives one hard path per task. A *kill row* is the usage row with the latest start time at or before each FAIL or KILL event of a task. About 80 percent of kill rows decode to state 1, whose means on the CPU features sit close to where a raw zero lands after the log transform. The write-up calls state 1 a degraded or stalled regime and the state most associated with failure.

**Side experiments.** `notebooks/MCMC.ipynb` writes the forward algorithm inside the NumPyro model, so the hidden states are summed out and NUTS samples only the continuous parameters; on 20 tasks of 100 windows and four features it converges in under two minutes. `notebooks/Transformer.ipynb` trains a small transformer to predict a task's state 50 windows ahead, using row-wise state labels from an earlier version of the model. Neither appears in the write-up beyond the file listing.

The staff section below explains why the fitted model behaves as a mixture model rather than a sequence model, why the kill-row result is close to circular, what the training loop actually saw, and what the plan promised that was not built.

## Team and contributions

Team "Optimus Primes", Winter 2026.

- **Dhawal Dixit** ([@djdhawal](https://github.com/djdhawal)): the repository and its first three weeks: the DuckDB loader that reads trace shards from Cloud Storage in parallel, the feature-engineering class, the first NumPyro HMM in `src/modelling.py` (a `scan`-based model in the style of NumPyro's HMM example), `notebooks/pull_data.ipynb`, `notebooks/hmm_model_v01.ipynb`, and the training-loss figure.
- **Jenitha Patel** ([@jenithapatel](https://github.com/jenithapatel)): the final pipeline in `notebooks/hmm_model_v02.ipynb`, with the train/test split, the kill-row matching, the SVI fit, the NumPy forward filter and Viterbi, and the state analysis; `notebooks/validation.ipynb`; both saved parameter files; the four figures the write-up uses; the `src/` layout; and most of the write-up.
- **Sameeksha Kini** ([@SameekshaKini](https://github.com/SameekshaKini)): `src/hmm_decoder.py`, an edit to the feature-engineering module, and the write-up's section on SVI.
- **RJ Xia** ([@rjxia993](https://github.com/rjxia993)): `notebooks/MCMC.ipynb` and `notebooks/Transformer.ipynb`.
- **Sabayna Ali** ([@sabayna-ali](https://github.com/sabayna-ali)): the first README, on February 26.

The repository ran from February 16 to March 13, 2026. The notebooks were saved from Colab and Kaggle, so the account that committed a notebook is not necessarily its only author; the attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/wi26-borg-failure-hmm/commits/main).

## Getting started

The authors' data extract is not in the repository; the notebooks read it from a Google Drive folder. [data/README.md](data/README.md) explains how it was made and how to rebuild it. The saved model parameters are in the repository, so the decoding and validation steps and the staff check below run without the trace.

`requirements.txt` pins DuckDB, JAX, NumPy, NumPyro, pandas and regex. The notebooks also use PyArrow, SciPy, scikit-learn, seaborn, matplotlib, Optax and funsor, which NumPyro's enumeration needs; the Transformer notebook needs PyTorch.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pyarrow scipy scikit-learn seaborn matplotlib optax funsor jupyter
```

**The pipeline.**

1. `notebooks/pull_data.ipynb` samples tasks and pulls their usage rows from the public bucket. See the data note for what its recorded run covered.
2. `notebooks/hmm_model_v02.ipynb` is the whole project in one notebook: cleaning, labels, the split, features, the SVI fit, decoding, the kill-row analysis and the figures. Set `path` in the first cells to the folder holding the two Parquet extracts.
3. `notebooks/validation.ipynb` reloads `data/hmm_model_march11.pkl` and `data/norm_params.pkl` and repeats the decoding and the kill-row tables over the whole extract.

**Staff check.** `staff/check_row_argmax.py` loads the saved parameters with an unpickler that admits only NumPy types, prints the transition matrix and the state profiles, and simulates sequences from the fitted HMM to measure how often the forward filter and Viterbi disagree with a row-by-row classification that ignores the dynamics. It needs only NumPy:

```sh
python staff/check_row_argmax.py
```

**What does not run as committed.** Nothing under `src/` is imported by any notebook; the notebooks carry their own copies of the same functions. `src/hmm_decoder.py` has an indentation error at line 147 and its methods lack `self` and imports, so it cannot be imported. In `src/data_cleaning.py` the number pattern is a plain string rather than a compiled regex, so the percentile parser fails, and two methods lack `self`. `notebooks/MCMC.ipynb` uses names (`iu_sorted`, `mcmc_mu`, `colors`) that no surviving cell defines, so it needs small repairs before it runs top to bottom. `notebooks/hmm_model_v01.ipynb` is the abandoned first pipeline.

## Data

The trace extract is never committed; `.gitignore` excludes Parquet files and everything in `data/` except the two parameter files and the data note.

- **Source:** [Google ClusterData2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) (Tirmazi et al., "Borg: the Next Generation", EuroSys 2020), one month of traces from eight Borg cells. The team used cell *a* and three tables: `instance_usage` (resource measurements in five-minute windows; more than 12,000 Parquet shards for this cell), `instance_events` (lifecycle events; EVICT is type 4, FAIL 5, KILL 7) and `collection_events` (job metadata including priority).
- **Terms:** CC BY 4.0. Cite the trace and the paper.
- **The extract:** 1,200 jobs with a priority bin and an outcome, from a table built outside the repository; lifecycle events for those jobs from the first ten event shards; a stratified sample of up to 100 tasks per outcome and priority bin, 905 tasks; and their usage rows. The recorded run of the pull notebook reads 500 of the usage shards and finds rows for 712 tasks; the extract the modelling notebooks load has all 905 tasks and 3,436,845 rows, so a larger pull was made later and not recorded. Because shards are not organized by task or time, a partial pull leaves each task a sparse sample of its windows: in the recorded run the median task had about 94 minutes between consecutive rows.
- **Labels:** last event FAIL or KILL, 140 of 905 tasks. Kill rows: 4,036 of 3.25 million rows, 3,041 in the training tasks and 995 in the test tasks. A user's kill, a cleanup kill by a parent job and a system kill are not told apart.
- **Crash loops:** the tasks with the most rows (97,593, 67,259 and 66,760 at the top) all ended in FAIL or KILL and have rows seconds apart, tasks restarted again and again. A few of them hold a large share of all rows.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: problem, dataset, pipeline, model, interpretation, validation, limitations, future work. |
| `notebooks/pull_data.ipynb` | Sampling tasks by priority bin and outcome and pulling their events and usage rows from the bucket with DuckDB. |
| `notebooks/hmm_model_v02.ipynb` | The final pipeline: cleaning, labels, split, features, the NumPyro SVI fit, forward filter and Viterbi, kill-row analysis, figures, saving the parameters. |
| `notebooks/validation.ipynb` | Reloads the saved parameters and repeats the decoding and kill-row tables over the whole extract. |
| `notebooks/hmm_model_v01.ipynb` | The first pipeline, built on `src/modelling.py`: 200-window sequences, 15,000 SVI steps. Abandoned after it collapsed to one state. |
| `notebooks/MCMC.ipynb` | The marginalized HMM sampled with NUTS on 20 tasks. |
| `notebooks/Transformer.ipynb` | A transformer that forecasts row-wise state labels 50 windows ahead, run on Kaggle. |
| `src/data_loader.py` | The parallel DuckDB loader for the trace shards, as a class. |
| `src/data_cleaning.py` | Feature engineering: percentile parsing, burstiness, tail features, log-normalization. Not importable as is. |
| `src/modelling.py` | The first HMM, a `scan`-based NumPyro model with enumeration, and its SVI setup. |
| `src/hmm_decoder.py` | Forward filter and Viterbi as a class. Does not parse. |
| `data/hmm_model_march11.pkl`, `data/norm_params.pkl` | The saved point estimates and the normalization statistics. See `data/README.md`. |
| `figures/` | The feature distributions against the prior, the feature correlation heatmap, average usage by decoded state, transitions into kill rows, and the training loss of an early run of the first pipeline. |
| `staff/check_row_argmax.py` | Course staff's check of how much the dynamics matter, described above. |

## Method

**Preprocessing.** Usage rows are de-duplicated on task and start time, and events on task and event time. The CPU usage percentile distribution of each window is parsed into eleven percentile columns and a burstiness column; the tail distribution into mean, maximum, 90th percentile and non-zero fraction. Each task's last event sets its label. Tasks are split 80/20, stratified on the label, and the split is checked for overlap. Eleven usage columns are filled with zero where missing, log-transformed with an offset of 10⁻⁷ and standardized with the training tasks' mean and standard deviation; five of them are the model's features. Each task's rows are sorted by start time into one sequence.

**Model.** For a task with windows *t* = 1, …, *T* and five-dimensional observations *x*ₜ,

> *z*₁ ~ Categorical(π), *z*ₜ | *z*ₜ₋₁ ~ Categorical(A[*z*ₜ₋₁]), *x*ₜ | *z*ₜ ~ Normal(μ[*z*ₜ], σ[*z*ₜ]) independently per feature,

with π ~ Dirichlet(1, 1, 1), each row of A ~ Dirichlet(1, 1, 1), μ ~ Normal(0, 1) and σ ~ Exponential(1) elementwise. The number of states is fixed at three.

**Inference.** The hidden states are declared with `infer={"enumerate": "parallel"}` and hidden from the guide with `handlers.block`, so `TraceEnum_ELBO` sums them out exactly at every step while an `AutoNormal` guide places independent Gaussians on π, A, μ and σ in unconstrained space. Each step draws four training tasks at random, truncates each to its first ten windows, pads shorter ones and masks the padding. Optax's Adam at 0.005 with the rate halving every 1,000 steps and gradients clipped at global norm 5 runs for 1,000 steps; the loss on a batch falls from about 310 to between 165 and 180 and is noisy throughout. Afterwards 200 draws from the guide are averaged to point estimates. The notebook's own timing note says 21 minutes.

**Decoding.** The forward filter runs in log space with normalization at every step and returns P(*z*ₜ | *x*₁..ₜ); its argmax is the "dominant forward state". Viterbi returns the most likely path. Both are plain NumPy and are applied to every task's full sequence of rows, except that the training-set state summary in the notebook caps sequences at 500 windows.

**The NUTS variant.** The same emissions, a sticky Dirichlet(0.9 I + 0.1) prior on the transition rows and no initial distribution, with the forward algorithm written in `lax.scan` inside the model and the summed log-likelihood added through `numpyro.factor`. Twenty tasks with at least 100 windows, their first 100 windows, four raw usage features standardized per task. NUTS with 500 warmup and 1,000 draws, one chain, runs in about a minute and a half on CPU with R-hat 1.00 and effective sample sizes near 900. The notebook notes that 64 tasks gave R-hat above 1.1.

**The transformer.** State labels come from an earlier eleven-feature model dated March 8, assigned row by row as the state with the highest emission likelihood, with no dynamics. A two-layer encoder with 64-dimensional embeddings and four heads reads 50 windows and predicts the label 50 windows later. It trains for 30 epochs on a 20 percent subsample, 661,268 training windows and 19,203 test windows from a task-level 80/20 split.

### What is Bayesian here

The HMM puts priors on every continuous parameter, and the hidden states are integrated out exactly: by enumeration in the SVI model, by the forward algorithm written into the NUTS model. That is the evaluation problem of the course's HMM sessions, implemented twice. Stochastic variational inference with a mean-field Gaussian guide then approximates the posterior over π, A, μ and σ by maximizing the evidence lower bound, and NUTS samples the same kind of posterior in the second notebook. The forward filter and Viterbi used for decoding are the filtering and decoding problems of the course, run with fixed parameters.

What stops short of the plan: the posterior is collapsed to its mean right after fitting, so none of the decoded probabilities, kill-row tables or figures carries any parameter uncertainty, and the filtered probabilities are uncertainty about the state given one fixed parameter set. The methodology document promised credible intervals that would tell "migrate the task now" from "wait and watch"; no interval appears in the results. The mini-batch objective is not scaled up to the size of the data, so the guide approximates the posterior one would get from 40 windows, with the priors weighing accordingly. There is no posterior predictive check and no held-out likelihood, and the planned comparison of two to five states by ELBO and WAIC was not run. The NUTS notebook is the one place where posterior spread is reported, as 5 and 95 percent quantiles of the state means.

## Results

**The fitted model.** From `data/hmm_model_march11.pkl`. The initial distribution is (0.30, 0.30, 0.40).

| From \ to | State 0 | State 1 | State 2 |
|---|---:|---:|---:|
| State 0 | 0.29 | 0.29 | 0.42 |
| State 1 | 0.19 | 0.41 | 0.40 |
| State 2 | 0.11 | 0.14 | 0.75 |

Expected dwell times are 1.4, 1.7 and 4.1 windows; the stationary distribution is (0.15, 0.22, 0.62).

**State profiles.** Emission means in standardized log units, with the value a raw zero takes after the transform for comparison.

| Feature | State 0 | State 1 | State 2 | Raw zero |
|---|---:|---:|---:|---:|
| Average CPU | −0.69 | −0.94 | 0.74 | −1.44 |
| CPU 90th percentile | −0.58 | −1.06 | 0.79 | −1.34 |
| Average memory | −0.67 | −0.73 | 0.74 | −2.46 |
| Assigned memory | 0.03 | −0.08 | 0.45 | −5.92 |
| CPU burstiness | −0.67 | −1.12 | 0.82 | −1.23 |

State 2 has scales of 0.36 to 0.66; state 1 has 0.46 to 1.03; state 0 has 0.76 to 1.27 and overlaps state 1 heavily.

**Kill rows by decoded state.** The write-up's tables come from `validation.ipynb`, which decodes the whole extract, training tasks included. The training notebook gives the split numbers.

| Share of kill rows in … | State 0 | State 1 | State 2 |
|---|---:|---:|---:|
| Whole extract, forward state | 12.9% | 78.0% | 9.1% |
| Whole extract, Viterbi | 10.3% | 80.8% | 8.9% |
| Training tasks, forward state | 14.3% | 75.8% | 9.9% |
| Test tasks, forward state | 8.9% | 85.1% | 5.9% |
| Test tasks, Viterbi | 5.4% | 88.7% | 5.8% |

**What the write-up leaves out.** Both notebooks also compute the base rates and the per-row kill rate; whole extract, forward state.

| | State 0 | State 1 | State 2 |
|---|---:|---:|---:|
| Share of all rows | 8.2% | 39.3% | 52.4% |
| Share of kill rows ÷ share of all rows | 1.57 | 1.98 | 0.17 |
| Kill rows per 1,000 rows in the state | 1.8 | 2.3 | 0.2 |

In the window before a kill, the Viterbi state on the training tasks is 0, 1 or 2 for 10.5, 65.7 and 23.8 percent of kills, against 8.7, 35.9 and 55.4 percent of all rows. Of kills whose previous window was in state 2, 78 percent are in state 1 at the kill row itself.

**NUTS.** Posterior means of the four state means are tightly estimated (standard deviations of 0.01 to 0.04 on standardized features) and the chain is clean. The figure of state separation was saved to the authors' Drive and is not in the repository.

**Transformer.** Accuracy 0.62 on the test windows, of which 58 percent carry the majority label. Recall is 0.99 on state 1 and 0.09 and 0.12 on the other two: the model predicts state 1 almost regardless of input. The saved results file lists 15 epochs; the training loop ran 30.

**The first pipeline.** `hmm_model_v01.ipynb` fit the `src/modelling.py` model on 76 windows of 200 rows for 15,000 steps. It put 96 percent of rows in one state and learned three identical transition rows, which is why it was abandoned. The committed training-loss figure, 15,000 steps from 1.8 × 10⁷ down to 1.7 × 10⁶, is from a run of that pipeline on March 3, not of the final model.

## What to learn from this project

Course staff added this section. The pipeline is a substantial piece of work; the points below are about what its results can and cannot support.

- **Check whether your sequence model is using the sequence.** The fitted transition matrix is barely sticky: 0.29, 0.41 and 0.75 on the diagonal, so a task is expected to stay in a state for one to four windows. Meanwhile the emission log-likelihood of a row differs between states by tens of nats, while the largest transition penalty is 2.2 nats. The filtered state at a row is therefore the state whose emission distribution fits that row, and the dynamics almost never override it. On sequences simulated from the fitted model itself, the staff script finds the forward filter agreeing with a row-by-row classification on 98 percent of rows and Viterbi on 98 percent; with the same emissions and a sticky transition matrix, Viterbi recovers 98 percent of the simulated states where the row-by-row classification gets 87 percent, which is what a sequence model earning its keep looks like. A three-component Gaussian mixture would have produced the same states. The fix is not to drop the HMM but to look at its transition matrix before interpreting the states as regimes, and to compare it with the mixture.
- **Know what your states are.** State 1's means on the three CPU features sit a fraction of a standard deviation above where a raw zero lands after log(x + 10⁻⁷), with assigned memory near average. State 1 is "the CPU counters read zero or nearly so while memory stays assigned", and the spike at that position is visible in the write-up's own histogram figure. The authors' reading, resources allocated but idle, is right; "a degraded regime the model discovered" overstates what a mixture component on log-usage can know.
- **Beware validation that scores the label's own footprint.** A kill row is the window in which the task died, and usage in that window is low because the task stopped. That it decodes to the low-usage state is nearly guaranteed by the definition. The transitions figure shows it: of kills reached from the high-usage state, 78 percent are in state 1 at the kill row, so the state flips at the moment of death rather than before it. Prediction means scoring the state at a fixed lead time before the event, on held-out tasks, against an honest baseline such as "is CPU usage zero".
- **Report the numbers that cut against the headline.** The notebooks compute the base rates, the lift and the per-row kill rate; the write-up shows only the share of kill rows per state. State 1 holds 39 percent of all rows, so its 78 percent of kill rows is a lift of about two; state 0 has a lift of 1.6 and a per-row kill rate within a fifth of state 1's. The defensible finding is the negative one: kills rarely coincide with high-usage rows.
- **Say what the training loop saw.** Batches of four tasks, truncated to their first ten windows, for 1,000 steps: about 7,000 rows out of 2.5 million, all from the first 50 minutes of each task's record, and the mini-batch loss never scaled up to the data. Decoding is then applied to sequences up to 97,593 rows long. Mini-batching an HMM is fine; truncation to the start of every task is a modeling choice, and the write-up presents it as a JAX implementation detail.
- **Look at the time axis.** The trace measures every five minutes; the recorded pull takes a sparse sample of the shards, and the modelling notebook shows a task whose rows are 25 seconds apart. A single transition matrix over rows whose spacing ranges from seconds to hours has no unit of time. Resample to a grid, or let the transition depend on the gap.
- **Ship the plan or revise it in writing.** The methodology document promised per-program transition matrices pooled by priority tier, a cycles-per-instruction feature, choosing the number of states by ELBO and WAIC, calibration metrics, and separating parent-cleanup kills from failures. None of it is in the repository, and the repository's description still claimed the hierarchy and the WAIC comparison. The limitations section does admit the label mixing, which is the honest way to handle a dropped plan.
- **Keep the repository honest and runnable.** The `src/` modules are unused and partly broken, and the write-up calls them "modularized helper code". Four "INSERT FIGURE HERE" comments survive in the README, "300,000 timesteps" where the longest task has 97,593 rows, "regular time intervals", and `norms_params.pkl`. The validation notebook decodes all 905 tasks, training tasks included, and the write-up calls its tables validation. The NUTS notebook, which is the right method for this question at the scale it was run, gets one line in the file listing.

What the project does well: a real production trace and the engineering to work with it, a parallel DuckDB loader over thousands of remote shards and memory-aware Parquet reading; an HMM with exact enumeration inside stochastic variational inference, which few students get working; a forward filter and Viterbi written by hand and correct; a task-level stratified split with a check for leakage; a marginalized HMM under NUTS with convergence diagnostics and a note on when it broke; a long, well-organized write-up with a genuine limitations section; and the question itself, which deserves the better evaluation described above.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Removed `.DS_Store` and `.vscode/settings.json` and added a `.gitignore` that keeps the trace extract out of the repository while keeping the two parameter files.
- Added `data/README.md`, on how the extract was built and how to rebuild it, and `staff/check_row_argmax.py`, the check described above.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The Borg trace is CC BY 4.0, as above.
