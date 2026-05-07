# SageMaker AI — Classroom demos

Two Jupyter notebooks, each a self-contained 90-minute classroom session against real AWS resources:

| Notebook | What it covers |
|---|---|
| [`sagemaker_comprehensive_demo.ipynb`](sagemaker_comprehensive_demo.ipynb) | Every major SageMaker AI building block end-to-end: Processing → Training → Pipelines → Model Registry → Real-time Endpoints (with DataCapture) → Auto-scaling → Model Cards → Model Monitor. |
| [`sagemaker_mlops_demo.ipynb`](sagemaker_mlops_demo.ipynb) | The MLOps surfaces: Feature Store (online + offline + Athena) → Experiments → Hyperparameter Tuning (Bayesian) → Pipelines with `ConditionStep` → **Model Registry approval workflow with EventBridge → Lambda auto-deploy** → Lineage Tracking. |

**Audience:** intermediate AWS / ML engineers who want to see SageMaker workflows end-to-end with real (small, quota-safe) AWS resources.

---

## MLOps demo (`sagemaker_mlops_demo.ipynb`)

| § | Topic | Cell time |
|---|---|---|
| 0 | Resilient setup (boto3 + STS, defensive across SageMaker Distribution image versions) | 1 min |
| 1 | SDK install / version pin (`sagemaker>=2.246.0,<3.0`, `pyathena`) | 1 min |
| 2 | California Housing dataset → S3 (training CSVs + Feature-Store-shaped DataFrame) | 1 min |
| 3 | **Feature Store** — `FeatureGroup` with online (DDB, sub-100ms) + offline (S3 + Glue + Athena) stores | 5–6 min |
| 4 | **Experiments** — three tracked training runs (`sagemaker.experiments.Run`) for side-by-side metric comparison | 8 min |
| 5 | **Hyperparameter Tuning** — `HyperparameterTuner(strategy="Bayesian")` over `eta` × `max_depth` × `subsample`; 4 trials, max 2 in parallel | 12–15 min |
| 6 | **Pipeline** — `Preprocess → Train → Eval → ConditionStep` that branches on RMSE; passes go to `RegisterModel(PendingManualApproval)`, fails go to `FailStep` | 10–12 min |
| 7 | **Model Registry approval → auto-deploy** — flip `ModelApprovalStatus="Approved"` → EventBridge rule fires → Lambda creates Model + EndpointConfig + Endpoint asynchronously → poll until `InService` → live invocation | 5–7 min |
| 8 | **Lineage Tracking** — query the artifact graph (endpoint → model package → training/processing jobs → input artifacts), plus an Athena query against the offline Feature Store | 1 min |
| 9 | Class discussion prompts | 5 min teaching |
| 10 | **Cleanup** — endpoint, package + group, pipeline, feature group, S3 sweep | 1 min |

Total: ~55 min cell execution + ~35 min teaching = **90 min**. Estimated cost: **~$2–3** at on-demand `us-east-1` pricing.

> **Prereq specific to this notebook:** the **EventBridge rule + auto-deploy Lambda** must already be deployed in the account. The CFN template that provisions them is kept private to the classroom (it hardcodes the model package group name `mlops-pkg-group` so other Registry approvals don't trigger deployments). If you fork this notebook, you'll need to wire up your own EventBridge → Lambda chain — the rule pattern is documented inline in §7.

> **Three SDK gotchas worked around in §4 + §5 + §8** (so you don't have to learn them the hard way):
> 1. `sagemaker.experiments.Run(run_name=...)` — no dots allowed in `run_name` (TrialComponent regex `[a-zA-Z0-9](-*[a-zA-Z0-9]){0,119}`). Encode floats as integers.
> 2. Built-in algorithm Estimators + `HyperparameterTuner` reject `metric_definitions=[...]` (`ValidationException: You can't override the metric definitions for Amazon SageMaker algorithms`). Drop it.
> 3. `LineageQuery.query()` requires `include_edges=True` or a Filter; `EndpointContext.load(endpoint_name=...)` doesn't exist — look up by `Context.list(source_uri=endpoint_arn)`; Vertex attrs are `lineage_entity` / `lineage_source` (not `..._type`).

---

## Comprehensive demo (`sagemaker_comprehensive_demo.ipynb`)

### What it covers

| § | Topic | Cell time |
|---|---|---|
| 0 | Resilient setup (boto3 + STS, defensive across SageMaker Distribution image versions) | 2 min |
| 1 | SDK install / version pin (`sagemaker>=2.246.0,<3.0`) | 1 min |
| 2 | California Housing dataset → S3 (75/15/10 train/val/test split) | 2 min |
| 3 | **Processing** with `SKLearnProcessor` (StandardScaler, train/val/test channels) | 6–8 min |
| 4 | **Training** with the built-in XGBoost container, metrics via `TrainingJobAnalytics` | 6–8 min |
| 5 | **Pipeline** chaining Processing → Training → RegisterModel, with step-level caching | 8–10 min cold / ~2 min cached |
| 6 | **Deployment** of the latest registered model package to a real-time endpoint, with `DataCaptureConfig` | 7–9 min |
| 7 | Live invocation + verification that captured payloads land in S3 | 1 min |
| 8 | **Auto-scaling** the endpoint with target-tracking on `InvocationsPerInstance` | 3 min |
| 9 | **Model Card** populated from the training job's analytics, surfaced in the Studio Model Dashboard | 5 min |
| 10 | **Model Monitor** baseline + hourly schedule against the live endpoint | 6–8 min |
| 11 | Class discussion prompts | 2 min teaching |
| 12 | **Cleanup** — endpoint, schedule, autoscaling target, model package + group, model card, pipeline, S3 prefix | 2 min |

Total: ~65 min cell execution + ~25 min teaching = **90 min**.

---

## Prerequisites

- An AWS account where you can launch SageMaker Studio (~$3 for one full demo run on default quotas).
- A SageMaker Studio domain in any region you have quota for (default: `us-east-1`).
- A SageMaker execution role with these AWS-managed policies attached (or equivalent inline):
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
  - inline policy granting `iam:PassRole` on **its own ARN** (critical — Pipeline / Training / Endpoint deploys fail without it; `AmazonSageMakerFullAccess` does not include this on custom roles)
- A JupyterLab space launched in the domain on `ml.m5.large` or larger.

### Quota notes (defaults, fresh AWS account)

The notebook intentionally uses three different instance types because the default service quotas in a brand-new AWS account are surprising:

| Workload | Instance type | Why this one |
|---|---|---|
| Processing job (§3) and Model Monitor (§10) | `ml.t3.large` | `ml.m5.large` for processing has quota **0** by default; `ml.t3.large` has quota **4** |
| Training job (§4) and pipeline training (§5) | `ml.m5.xlarge` | gated by the global "Number of instances across all training jobs = 4" |
| Real-time endpoint (§6) | `ml.m5.large` | endpoint quota for `ml.m5.large` is 4 by default |

If you've raised your quotas, you can swap up — search the notebook for `instance_type=` and adjust.

---

## How to run

1. Open SageMaker Studio → JupyterLab.
2. Upload `sagemaker_comprehensive_demo.ipynb` (or pull it from this repo via the JupyterLab terminal).
3. Pick the **`Python 3 (SageMaker Distribution)`** kernel (CPU image is fine).
4. Run cells in order, top to bottom. The first time `%pip install` upgrades the SDK, **restart the kernel** before continuing — there's a note in §1 about this.
5. **Always run §12 (Cleanup) before logging off.** The endpoint costs ~$0.10/hour while it's `InService`; the hourly Model Monitor schedule costs another ~$0.20/hour.

---

## Cost

On default quotas + on-demand pricing in `us-east-1` (May 2026):

- Full run including 12-section cleanup: **~$1–3** depending on how long you linger between sections.
- If you forget §12 cleanup and walk away for 12 hours: **~$3** (endpoint + monitoring schedule are the two ongoing costs; everything else is metered to the second).
- If you only delete the endpoint but leave the JupyterLab space running for 12h: **~$1.45**.

The notebook stores all artifacts under one S3 prefix (`comprehensive-demo/<RUN_ID>/`) so cleanup-by-prefix in §12 is trivial.

---

## What you'll see in the Studio UI

- **Pipelines** view: the 3-step DAG (`PreprocessCaliforniaHousing` → `TrainXGBoost` → `RegisterModelPackage`) with per-step status and cache hits highlighted.
- **Model Registry**: the registered model package with `Approved` status (set programmatically in §5).
- **Endpoints**: the live `cal-housing-ep-<RUN_ID>` endpoint with autoscaling policy attached.
- **Model Dashboard → Model Cards**: the populated Model Card with intended use, training metrics, ethical considerations, and hyperparameters.
- **Model Monitor**: the hourly schedule (in `Scheduled` state) and, after the top of the next hour, drift reports comparing live captures to the suggested baseline.

---

## Class discussion prompts (from §11)

1. **Pipelines vs. notebooks** — what changes when you have to retrain weekly with the same code path? Why is the pipeline cache load-bearing?
2. **DataCapture sampling** — the demo uses 100% sampling. What's the cost trade-off in production?
3. **Model Card status** — the demo leaves it in `Draft`. Who should have permission to transition to `PendingReview` / `Approved`, and what gates would you put in front of that?
4. **Auto-scaling target value** — the demo uses `InvocationsPerInstance=20`. How would you pick this in real life?
5. **Monitor → action** — what should happen when the schedule reports a constraint violation? Page on-call? Block deployments? Open a ticket?

---

## Known SDK pitfalls (already worked around in this notebook)

- PyPI `sagemaker` v3.x is a **different package** with a near-empty API. `pip install sagemaker>=2.232.0` resolves to v3.x. The notebook pins `>=2.246.0,<3.0`.
- `ProcessingInput.LocalPath` must be unique across inputs — the notebook uploads all raw CSVs to one S3 prefix and passes a single `ProcessingInput` for the prefix, not three.
- `ModelPackage.deploy()` returns `None` (not a `Predictor`) when the package has no `predictor_cls`. The notebook constructs a `Predictor` explicitly after deploy.
- `Estimator(image_uri=image_uris.retrieve("xgboost", ...), metric_definitions=[...])` fails — built-in algorithm containers reject custom metric definitions. The notebook reads metrics via `TrainingJobAnalytics` instead.
- Model Card schema (`sagemaker.model_card`) requires nested types (`ObjectiveFunction(function=Function(function=ObjectiveFunctionEnum.MINIMIZE, facet=FacetEnum.RMSE, ...))`) and rejects free-form strings for `risk_rating` (must be one of `High` / `Medium` / `Low` / `Unknown`).

---

## License

MIT — feel free to use, modify, and redistribute for classroom or production use. Attribution appreciated but not required.
