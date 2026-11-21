# Model contracts and evidence

These provider-owned records use [MCEG](https://github.com/HajimeKawahara/mceg)
to connect model definitions, code, evidence, scoped Assessments and Decisions.
`escape` does not import or depend on MCEG. Use a separate authoring environment;
the scientific package and its ordinary tests work without MCEG.

## Current status

`supported`, `contradicted` and `unknown` describe an evidence-support Assessment
for one Claim revision, implementation/data Subject and Context. A numerical
failure is a contradiction only when a valid comparison violates that Claim's
criterion. A crash, missing artifact or untested physical proposition remains
inconclusive. Historical judgments retain their original records.

| Model or claim | Current recorded finding | Scope and remaining work |
| --- | --- | --- |
| Isothermal/nonisothermal Parker wind, with or without linear tides | Formulation and sampled numerical checks are `supported` | Prescribed temperature and constant mean molecular weight; no energy equation or planetary accuracy claim. See the [wind contract](contracts/escape.parker_wind.yaml). |
| Six-state local hydrogen NLTE and H-alpha transfer | Historical formulation/numerical reviews are `supported` | Their exact source and test scopes are retained; atmospheric accuracy and posterior convergence are `unknown`. See the [local NLTE contract](contracts/escape.hydrogen_nlte.yaml). |
| Coupled radiation, populations, temperature and finite flow | Scoped numerical reviews are `supported` | Finite boundary problems; transonic selection, moving-line transport and planetary accuracy remain unverified. |
| Earlier nmax=4/5 finite atom versus Cloudy | `supported` in the eight declared cases | Maximum n2/n3 difference **1.100%**, below **5%**; reference refinement changes them by less than **0.049%**, below **1%**. |
| Original finite-atom agreement with Cloudy | New reviewed Assessments are `contradicted` | Original n3 errors **32.65%** and **30.22%** exceed **5%**. Original `unknown` records remain as history. |
| Original and repaired atom-size convergence | New reviewed Assessments are `contradicted` | nmax=4 to 5 changes n3 by **11.46%** in the original atom and **7.735%** in the repaired atom, exceeding **5%**. These original finite-atom claims remain false; the enlarged model has separate criteria below. |
| Enlarged hydrogen atom (default nmax=15, supported through 20) | New convergence and reference reviews are `supported` | Baseline n3 changes **0.2335%** from 15 to 20; maximum n2/n3 change across seven conditions is **1.4713%**. The fourteen matched Cloudy comparisons differ by at most **2.0315%**, below the unchanged **5%** criterion. |
| Complete spectrum families and NIST/Rydberg center choices | Six inventory entries remain without dedicated contracts | Existing population or wind checks do not validate their complete geometry, transfer, broadening or observational applicability. |

The [Cloudy investigation contract](contracts/escape.cloudy_investigation.yaml)
connects the original failure, paired rate-block experiments, physical choices
and repaired comparisons. The [coupled contract](contracts/escape.coupled_hydrogen.yaml)
retains the earlier formulation and execution reviews. The new contradiction
reviews use the measured criteria, not a reinterpretation of all failed tests.

The [enlarged-atom contract](contracts/escape.atom_convergence.yaml) records why
high-shell recombination and cascades change the low populations. It adds explicit
shells through n=20, chooses nmax=15 as the default and retains the original
4-to-5 contradiction. The nmax=10 candidate failed the dilute Cloudy convergence
check (6.96%); it is not silently omitted. Independent n=30/40 reference tails
and continuum refinement provide separate controls. These finite, sampled tests
do not establish infinite-atom or atmospheric accuracy.

The original baseline scope correction and finite-comparison Decision remain
unchanged. A new bridge Evidence and FollowUp resolution connect that historical
investigation to the enlarged-model result, without declaring its old Claim true.

Two diagnostic hypotheses still have `unknown` support Assessments even though
their prose records contrary indications: their cited numerical experiments
passed, and a separate failed hypothesis criterion has not been recorded.
They are not mechanically converted to `contradicted`. Whole-atmosphere physical
validation and the observational NUTS analysis also remain `unknown`.

The [inventory](models.yaml) covers 16 source/license files and 14 model entries.
Eight entries now link to contracts; six spectrum/line-center entries remain
uncontracted. All 14 model-boundary reviews remain pending human review.
`mceg coverage .` therefore still exits **1**; this is an explicit registration
and review gap, not evidence that the models are physically false.
The [earlier repository review](evidence/repository_review_20261121/receipt.json)
retains its original scope. Current enlarged-atom evidence is in the
[new review](evidence/atom_convergence_20261121/review.json).

## Equations and code correspondence

| Definition | Human-readable model | MCEG record |
| --- | --- | --- |
| Enlarged hydrogen atom and shell-convergence investigation | [Atom convergence PDF](../doc/atom_convergence.pdf) | [New contract](contracts/escape.atom_convergence.yaml) |
| Parker wind, prescribed nonisothermal temperature and linear tides | [Parker wind PDF](../doc/parker_wind.pdf) | [Equations and Bindings](contracts/escape.parker_wind.yaml) |
| Local six-state NLTE and H-alpha branch weights | [Local NLTE PDF](../doc/hydrogen_nlte.pdf) | [Historical local contract](contracts/escape.hydrogen_nlte.yaml) |
| Radiation, thermal balance and finite reactive flow | [Coupled-model PDF](../doc/coupled_model.pdf) | [Coupled contract](contracts/escape.coupled_hydrogen.yaml) |
| Repaired atomic rates and the Cloudy investigation | [Investigation PDF](../doc/cloudy_investigation.pdf) | [Investigation contract](contracts/escape.cloudy_investigation.yaml) |

The wind contract links momentum balance, the regular sonic point, the numerical
residual and implicit derivatives to the existing Python functions. Linear tides
enter the effective inward gravity as `GM/r^2 - tidal_coefficient_s2 * r`.
Temperature is supplied by the caller; this is distinct from the finite-flow
model that solves an energy equation.

Equation and Binding pins identify reviewed definitions and correspondences;
they do not prove mathematical equivalence. The historical PDFs and generated
appendices retain their original bytes and pins. In particular, the investigation
PDF's discussion of a future negative status describes the earlier MCEG reader;
the current vocabulary is documented above. Rebuilding an old document must not
silently renew its historical scientific review.

## Authoring environment and inspection

Use MCEG revision `7c004436e57617a63230afb7290ea04486ca5e96` or a compatible later
reader. This merged revision supports `contradicted` in schemas 0.2–0.4; older
readers reject the new value. Package and schema version numbers are independent.

```bash
python -m venv /tmp/escape-mceg
/tmp/escape-mceg/bin/python -m pip install 'mceg @ git+https://github.com/HajimeKawahara/mceg.git@7c004436e57617a63230afb7290ea04486ca5e96'
source /tmp/escape-mceg/bin/activate
```

From this provider checkout:

```bash
python -m mceg check
python -m mceg coverage .
python -m mceg check mceg/contracts/escape.parker_wind.yaml --assessment numerical_review --root .
python -m mceg check mceg/contracts/escape.atom_convergence.yaml --assessment convergence_review --root .
python -m mceg check mceg/contracts/escape.atom_convergence.yaml --assessment reference_review --root .
python -m mceg explain mceg/contracts/escape.cloudy_investigation.yaml --claim current_atom_truncation --investigation
python -m mceg equations mceg/contracts/escape.parker_wind.yaml --output doc/parker_wind_equations.tex --check
```

Selecting a `contradicted` Assessment makes its `recorded_support` check **FAIL**
and the command exit **1**. Inspect the named checks: that expected scientific
outcome is distinct from a broken artifact hash or stale Equation pin.
Without an Assessment selector, `check` lists historical judgments without
choosing one as the implicit current state. `UNKNOWN` alone may exit zero.
For the wind, incomplete external source traces and static inspection of JAX
decorators remain `UNKNOWN`; these are separate from the recorded test outcome.

The current source can be checked against the wind and enlarged-atom reviews.
The earlier investigation's `current_numerical_review` and reference/truncation
reviews now belong to historical source at `719ba25`, before the shell extension.
The older local/coupled reviews likewise belong to their recorded Subjects. To inspect the original coupled source, use its historical
checkout instead of refreshing old pins:

```bash
git worktree add --detach /tmp/escape-reviewed 6df616b
python -m mceg check mceg/contracts/escape.hydrogen_nlte.yaml --assessment numerical_review --root /tmp/escape-reviewed
```

The new contradicted reviews in the coupled contract also cite
`mceg/evidence/assessment_update_20261121/review.json`. To check those reviews
against the historical root, copy this one new review artifact into the same
relative path there. Preserve all historical source files and evidence.

The [historical replay](evidence/historical_replay_20261121/checks.json) and
[pre-optimization replay](evidence/historical_repaired_replay_20261121/check.json)
retain their exact checked scopes. The earlier
[completion result](evidence/cloudy_completion_review_20261121/readiness_after_review.json)
records readiness of the finite-atom comparison at that time; it is not a new
approval of every current follow-up or physical model.

## Evidence and repeatable verification

The new numerical gates retain the original failed `atom_truncation_accuracy`
gate and add separate enlarged-atom gates. Run the new verification with:

```bash
python tools/verify_coupled.py --gate enlarged_atom_convergence \
  --pythonpath /path/to/exojax/src --record results/atom-check-001
python tools/diagnose_atom_convergence.py --output results/atom-budget-001.json
```

The second command reconstructs the fixed-density Schur-complement attribution
and then solves charge conservation independently. The [new case manifest](investigations/atom_convergence_cases.json)
reuses the seven original unique conditions at nmax=15 and 20; its original 5%
criterion is unchanged. The [archive record](evidence/atom_convergence_20261121/archive.json)
identifies full raw executions stored in the separate evidence tag, including
failed intermediate attempts. Only compact records are kept in this branch.

The [wind execution](evidence/parker_wind_20261121/receipt.json) records the new
Parker/nonisothermal/tidal test run. The original
[local NLTE receipt](evidence/hydrogen_nlte_20261120/receipt.json) records 49
targeted tests. Its [example report](evidence/hydrogen_nlte_20261120/example.json)
also records a 32-warmup/32-sample synthetic NUTS smoke run: an execution check,
not a posterior-convergence claim.

The repaired [numerical receipt](evidence/coupled_repair_20261121_attempt03/receipt.json)
passes ten numerical gates and the baseline reference gate; the separate
atom-truncation gate fails. The [combined record](evidence/combined_verification_20261121/receipt.json)
accounts for 205 distinct tests (206 executions). These historical executions
are not silently reassigned to newly changed code. The
[optimized campaign audit](evidence/cloudy_campaign_audit_20261121_optimized/audit.json)
retains the eight-case comparison and reference-refinement results.

Full raw outputs and repeated source snapshots live in the immutable
`evidence/cloudy-20261121` tag. Follow the [retention policy](evidence/README.md)
and [restoration instructions](../doc/cloudy_reference.md) before replaying or
auditing those campaigns. Restore into a fresh ignored directory:

```bash
mkdir -p results/evidence-20261121
git archive evidence/cloudy-20261121 mceg/evidence | tar -x -C results/evidence-20261121
python tools/verify_cloudy_cases.py run --reference-batch results/evidence-20261121/mceg/evidence/cloudy_repair_cases_20261121_attempt02 --resolution .025 --record results/cloudy-replay-001
```

For a fresh external calculation, supply Cloudy explicitly as described in the
reproduction guide. [The case manifest](investigations/cloudy_validation_cases.json)
and [protocol](investigations/cloudy_discrepancy.json) define the original criteria.
Do not change a tolerance to turn a failed scientific claim into a pass.

The provider-owned [gate workflow](workflows/coupled.json) and
[runner](../tools/verify_coupled.py) execute tests and retain receipts. They never
edit Assessments, Decisions or definition pins:

```bash
python tools/verify_coupled.py --gate atom_truncation_accuracy
python tools/verify_coupled.py --gate atom_truncation_accuracy --pythonpath /path/to/exojax/src --record results/coupled-attempt-001
```

The first command displays the plan; the second runs prerequisite gates and the
criterion into a new directory. Numerical solving and physical adequacy remain
separate. A measured truncation failure can coexist with numerical conservation
and agreement with an equally truncated Cloudy atom.

## Maintaining the records

Preserve earlier Claims, Equations, Evidence and Assessments when revising their
meaning. Add a new scoped review and, when a choice is reconsidered, a new
Decision revision with `supersedes`. The original basis hashes should still
match the original records. Source drift or a changed definition requires review,
not a hash refresh presented as new scientific evidence.

Keep working outputs under ignored `results/` and retain only the compact
receipts and artifacts required by the reviewed claims. Contracts, documents
and compact evidence are included in source distributions, not wheels.
Installed-package discovery (`--package`) is not configured.

MCEG's native investigation view connects hypotheses, experiments, choices and
follow-ups, but it has no generic numerical `run` API or scheduler.
`FollowUp.requires` remains human-readable text. Execution policy and model
acceptance still require explicit scopes and criteria. Clear inventory
`review_pending` only after a maintainer reviews model boundaries and links.
