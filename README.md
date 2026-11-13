# ad-trajectory-modeling

Code for forecasting Alzheimer's disease (AD) dementia, cognitive decline and death from longitudinal clinic-visit records. The models are trained on ROSMAP (Rush Alzheimer's Disease Center) and tested externally on NACC.

**This is a code-only release.** It contains no data, no person-level outputs, no trained models and no results. ROSMAP and NACC data are available from RADC and NACC under their own data-use agreements. Code comments are mostly in Chinese. Comments that quoted results have been rewritten, so numbers in them are placeholders (`N`, `x.xx`). Cluster paths and account names are placeholders too.

## What the project does

One question runs through the whole project: **standing at a patient's visit, using only what is known up to that visit, what happens to this person over the next 1–10 years?**

The project is built around five parts:

1. **A unified landmark evaluation.** Every model, from logistic regression to a GPT-style health-trajectory model, answers the same questions on the same people. The same scoring code reads every model's answers.
2. **A model zoo.** About 50 models from 10 families. Models that can only predict directly get a *generative* version that simulates each person's future visit by visit. Only generative models enter the final comparison, so all of them can be compared on trajectories and perturbations, not just on point predictions.
3. **VISTA**, the selected model. It is a causal transformer over visits, trained only on next-step likelihood. Every multi-year answer comes from Monte-Carlo rollouts.
4. **A counterfactual perturbation framework.** It edits a person's record (genotype, cognition, comorbidity, age) and reads how the simulated future shifts. Expected directions are written down before any results are seen, and arms share random numbers.
5. **Extensions.** Genetics (single-locus variables plus pathway polygenic scores), gene × APOE and gene × comorbidity interaction tests, NACC pretraining and learning curves, and a pre-registered external validation on NACC.

## 1. Unified landmark evaluation (`unified_eval/`)

**Landmarks.** Every visit of every validation person is a landmark. A model sees the record up to and including that visit and nothing after it. People are split train / test / validation (`split811`). Train is used for fitting. Test is used for early stopping and hyperparameters. Validation is used only for reporting, and the same people are in validation for every tokenization and model.

**Questions.**
- *Part 1: prediction.* First occurrence within H years (H = 1, 2, 3, 5; main H = 2) of AD diagnosis, death, MMSE < 24, and global cognition < −1.
- *Part 2: checks for a perturbation engine.*
  - Simulation fidelity: simulated vs observed 10-year cumulative incidence and stage-transition matrices.
  - Perturbation plausibility: does risk move in the clinically expected direction?
- *Part 3: stage transitions.* Six "first crossing" questions over 1–10 years: MMSE < 27, MMSE < 24, cognition < 0, cognition < −1, AD, death.

**Only landmarks whose answer is known are counted.**
- A landmark is positive if the event happens inside the window.
- It is negative only if the person is observed past the window and has not crossed.
- Loss to follow-up inside the window and ambiguous first observations are excluded, so "not observed" is never scored as "did not happen".
- Death before crossing is handled two ways: excluded in the main analysis, and counted as negative in a competing-risk sensitivity analysis.

**Metrics** (`evaluate.py`, `evaluate_transitions.py`):
- AUC with 95% CIs from a 1000-draw person-level bootstrap. A person contributes several landmarks, so resampling landmarks would make intervals too narrow.
- Calibration: O/E by risk decile, and Brier skill.
- AUC within the same current stage, and within the same stage and 5-year age band, to separate real foresight from reading the current state.
- Sensitivity at 80% specificity for "transitioners vs stable".
- Time-to-event error against a constant guess.
- Three non-model reference rulers fitted on train: stage base rates, a stage + age logistic snapshot, and a continuous-score snapshot.

**How models answer.** Each model hands in one table: (person, landmark) → risk for each question at H = 1..10 years. The scoring code never looks inside a model.
- Generative models simulate 100 ten-year futures per landmark and count events, with smoothed risk (k + 0.5)/(n + 1).
- Delphi-style token models are read either through a query token placed one day after the landmark, using exponential-race rates, or through rollouts.

**Model selection** (`models_code/select_model.py`). Every model is compared with every other on 7 evaluation panels: prediction AUC, transition AUC, calibration, timing, 10-year incidence fidelity, transition-matrix fidelity, and perturbation direction. The tests use person-paired bootstrap differences with max-T simultaneous intervals. The decision rule ("tied best" vs "behind") was written before the run.

## 2. Model zoo (`models_code/`)

The registry is `models.py`, and the entry point is `run.py <name>`. All models share one input format, built by `input_data.py` and `input_data_trans.py`. Each visit has:
- 15 continuous measures, each with its own missingness mask;
- 9 states (prevalent conditions and medications), each with a missingness mask;
- "ever" flags, AD status, age, follow-up time and gap since the last visit;
- static background: sex, education, APOE, and optionally genetics.

| Family | Models |
|---|---|
| Landmark classifiers | logistic regression (demographics / current visit / hand-crafted history "L2C" features), random forest, XGBoost, LightGBM, TabPFN, feed-forward net |
| Risk scores | BDSI, ANU-ADRI, CAIDE |
| Landmark survival | Cox, Fine–Gray, random survival forest, XGBoost-Cox / -AFT, LightGBM-Cox, SurvivalBoost, DeepSurv, DeepHit, logistic-hazard, PMF, MTLR |
| Sequence models | GRU, LSTM, causal transformer, GRU-D, T-LSTM, RETAIN, Dynamic-DeepHit |
| Multistate | discrete Markov, covariate Markov, continuous-time multistate Markov (`ctmsm.py`, panel-data likelihood in torch) |
| Hidden Markov | HMM (`hmm.py`) |
| Joint models | two-stage LMM + hazard, shared-random-effects joint model, GPBoost / MERF / hierarchical-GP trajectories + hazard (`joint.py`) |
| Microsimulation | parametric g-formula with linear/logistic or LightGBM one-step models (`generators.py`) |
| Generative transformers | ETHOS (time-interval and quantile tokens, KV-cached generation; `ethos.py`), Delphi-2M adapted to ROSMAP tokenizations (`radc_delphi/`, `unified_eval/rollout_delphi.py`) |
| Generative RNN | MinimalRNN with model-based imputation (`generators.py`, `tune_minimalrnn.py`) |

**Making every model generative** (`rollout_versions.py`, `simulator.py`):
- A one-step model is either a direct-prediction model's own 1-year hazard (classifier, Cox/Breslow, or score) or a sequence net with an added next-visit head. It is paired with a model of the next visit's full distribution.
- `simulator.simulate_visits` then rolls each person forward year by year:
  - it samples death and AD, then continuous values, missingness and states;
  - prevalent conditions stay monotone;
  - the history features are replayed after each step.
- `full_draws=True` gives every trajectory its own random stream, so perturbation arms share random numbers (CRN) and differences between arms come only from the input edits.

**Delphi adaptation** (`radc_delphi/`):
- Delphi-2M is from gerstung-lab, used unmodified upstream.
- Several ROSMAP tokenizations: dedup, no-dedup, full-visit, snapshot.
- The rollout sampler adds:
  - absorbing death;
  - per-token "repeatable" rules, so ordinal bins and medications can recur;
  - a post-mortem block;
  - visit-size-aware sampling.
- Further variants: position-encoding leak checks, a static-only reference model, and pretraining on NACC.

## 3. VISTA, the selected model

VISTA is `transformer` in the code (`sequence.Causal`, `rollout_versions.SeqARModel`):
- It is a causal transformer whose tokens are visits.
- It is trained from scratch by maximum likelihood on the next step only: the 1-year hazard of death and AD, plus the full distribution of the next visit (continuous values, missingness, states). It uses teacher forcing and early stopping on test.
- It has no task heads, no reinforcement learning and no separate pretraining stage.
- Every answer at H years comes from rollouts.
- `rollout_versions.save_state` / `load_state` save a trained generator. A reload check requires identical outputs.

An improved variant is `transformer-gru-fill` (`models_code/trfx/`, config `s3-b-gru-fill.json`). It adds a 31-class MMSE head, a transformer→GRU hybrid encoder and MinimalRNN-style model-based imputation. It was selected on test only and is kept as supplementary.

## 4. Counterfactual perturbations

- **Definitions** (`unified_eval/perturbations.py`, `perturb_v2.py`, `perturb_features.py`):
  - APOE ε4 0→1; current MMSE set to 29/27/25/23; current cognition lowered; new stroke (now vs at the first visit); age +5 years; sex; education.
  - New-onset hypertension, diabetes, heart disease and depression, inserted in visit order.
  - Combinations guided by SHAP.
- **Translation per input format.** Each perturbation is translated into every model's own input: tokens for Delphi, features for the others. One perturbation therefore means the same clinical change for every model.
- **Pre-specified directions** are written before results are seen. A cell counts as right, wrong or "no response" (|log OR| < 0.05 or |ΔRMST| < 0.05 years).
- **Readouts:**
  - log OR and risk difference at 2, 5 and 10 years;
  - ΔRMST;
  - "years earlier": when the perturbed arm's mean cumulative-incidence curve reaches the reference arm's 10-year risk;
  - whether cognitive milestones move together with AD (synchrony).
- **Readout rules:**
  - only landmarks at risk at baseline are read;
  - person-level bootstrap CIs;
  - per-seed agreement across 5 retrained seeds;
  - a data-side reference (`perturb_reference.py`) puts the model's effect next to the adjusted observational association.
- **Sensitivity of the hypertension perturbation** (`experiments/genetics2/htn_probe/`). The probe also sets blood pressure and antihypertensive medication to a realistic new-onset profile, restricts to "clean" baselines, and decomposes the effect by variable.

## 5. Extensions (`experiments/`)

- **Genetics v1** (`models_code/geno_build.py`, `input_data_geno.py`): a lead-variant PRS from public GWAS stage-I summary statistics, ancestry PCs, and a has-genotype flag. A flag-only control separates cohort-entry effects from genetics.
- **Genetics v2** (`geno2_*.py`, `input_data_geno2.py`, `models_code/geno/`):
  - Tier 1: APOE genotype classes and BIN1 / CLU / PICALM / SORL1 lead variants as separate dosage variables. Imputation quality is checked, and LD proxies from the 1000 Genomes EUR panel are used where needed.
  - Tier 2: pathway PRS (immune/microglia, lipid, endocytosis, amyloid, tau, rest) from PRS-CS, with Tier-1 regions excluded. Pathway gene sets are pre-declared in `pathways_v1.json`.
  - Scores are residualized on PCs and z-scored on genotyped train.
  - A pre-declared non-inferiority margin on AUC, then gene perturbations at visits 2, 4 and 6, with a train-cohort sensitivity check.
- **Gene × APOE and gene × comorbidity** (`geno2_gxe.py`, `geno2_gxe_compare.py`):
  - 2 × 2 factorial perturbations under CRN.
  - Interaction is read on three scales: additive (interaction contrast), multiplicative (ratio of odds ratios) and time (years earlier). It is classified by pre-declared rules.
- **NACC.**
  - `nacc_data.py` harmonizes NACC UDS into the same input format. A self-check reproduces ROSMAP labels cell for cell.
  - Pretrain → fine-tune for VISTA and other architectures (`pretrain.py`, `nacc_delphi.py`, `delphi_upstream.py`).
  - Learning curves by training-set size (`experiments/nacc_scale/`, `experiments/nacc_scaling_ar/`).
  - A pre-registered external validation (`experiments/nacc_external/`): models are frozen on ROSMAP, a seal record is committed before unblinding, there is one unblinded run, and overlap with ROSMAP participants is checked.

## 6. Engineering

- Training and simulation run as Slurm jobs on an HPC cluster (`models_code/ris/`, `unified_eval/ris/`). The jobs are designed to fit 30-minute queues:
  - batches are trimmed and bucketed by length;
  - seeds run as spawned processes sized to the CPU affinity;
  - each epoch is checkpointed with a lock file;
  - follow-up jobs are chained with `--dependency=afterany`.
- **Data governance.** Person-level files (scores, trajectories, perturbation records) stay in a 0700 run directory on the cluster. Deploy and fetch scripts move only code one way and aggregate JSON the other way.
- **Reproducibility checks:**
  - saved and reloaded models must give identical outputs;
  - re-running an arm must give bit-identical risks;
  - truth tables are cross-checked cell by cell between Parts 1 and 3;
  - every Delphi query-token logit is checked against the scoring tables.

## Repository map

| Path | Contents |
|---|---|
| `models_code/` | model implementations, shared inputs, simulator, perturbation records, genetics, NACC, model selection, Slurm scripts (`ris/`) |
| `models_code/radc_delphi/` | Delphi vocabulary, `.bin` batching and rollout engine (upstream `model.py` / `utils.py` are not vendored) |
| `models_code/trfx/` | the transformer-gru-fill variant |
| `models_code/geno/` | pathway gene-set definition; lead variants from public GWAS summary statistics |
| `unified_eval/` | truth tables, scoring, rollouts, perturbation framework, SHAP, timing, sensitivity, figures |
| `experiments/` | genetics v2 + hypertension probe, NACC external validation, NACC learning curves |

## Running

The code expects:
- environment variables `AD_BASE` (data root) and `ZOO_RUN_DIR` (run directory);
- data files built from your own RADC / NACC data access.

Typical flow:
1. Build the inputs (`input_data.py`, `input_data_trans.py`) and the truth tables (`unified_eval/truth.py`, `transition_truth.py`).
2. Train and score a model with `python models_code/run.py <name>`.
3. Run perturbations with `unified_eval/perturb_v2.py`.
4. Aggregate with `models_code/compare.py` and `select_model.py`.

Python 3 with PyTorch, scikit-learn, XGBoost, LightGBM, pycox, and TabPFN for the models that use them.
