# Evidence retention

The subsequent enlarged-atom investigation has its own immutable tag,
`evidence/atom-convergence-20261121`. Its
[archive record](atom_convergence_20261121/archive.json) pins the commit and
archive hash. The 1.31 MB archive retains 35 fresh Cloudy executions, 14 matched
escape comparisons, numerical workflow attempts and causal diagnostics. Compact
reviews and test results remain in this tree. Restore it explicitly:

```bash
git show evidence/atom-convergence-20261121:mceg/evidence/atom_convergence_20261121/raw.tar.gz > /tmp/atom-convergence-raw.tar.gz
mkdir -p results/atom-evidence
tar -xzf /tmp/atom-convergence-raw.tar.gz -C results/atom-evidence
python tools/verify_cloudy_cases.py run \
  --cases mceg/investigations/atom_convergence_cases.json \
  --reference-batch results/atom-evidence/cloudy \
  --record results/atom-replay-001
```

The archived `archive-manifest.json` lists every original file hash. A replay
evaluates the current escape code against the retained reference; it does not
silently reuse earlier escape populations or renew scientific Assessments.

This directory contains review records and selected artifacts. The complete
2026-11-21 investigation is preserved at Git tag `evidence/cloudy-20261121`,
commit `c75bc5b819febcf1b3bed232ccebd4a889b6d829`.

| Material | Current tree | Full historical tag |
| --- | --- | --- |
| MCEG-referenced artifacts and source snapshots | Retained at their recorded paths | Original bytes retained |
| Primary receipts, campaign summaries and diagnoses | Retained for inspection | Original bytes retained |
| Raw campaigns and repeated reference outputs | Only selected required artifacts | Complete outputs, including failed attempts |
| Repeated input snapshots and intermediate diagnostics | Only selected required artifacts | Complete recorded snapshots, including dirty source |

Receipts and scientific Assessments keep their identities and outcomes. A
recorded failure is not removed from the investigation merely because its raw
files no longer appear in the current tree. The archived snapshot supplies
the original bytes for historical reproduction; receipt hashes are not refreshed
to fit a newer implementation.

Follow the [restoration instructions](../../doc/cloudy_reference.md#restore-the-historical-evidence-when-needed)
to extract the tagged evidence under ignored `results/evidence-20261121/`.
Reference replay and full campaign audits still verify every artifact required
by their receipts. Missing or changed raw data fail those checks; a saved PASS
label is insufficient. Restoration uses local Git objects and never fetches
anything automatically.

Source distributions include these compact review records. A versioned Git
checkout is needed to retrieve the full historical evidence. The scientific
package has no runtime dependency on Git, MCEG or Cloudy.

For new investigations, keep working outputs under ignored `results/`. Retain
the inputs, receipts and artifacts needed for the specific reviewed claims;
do not add an entire scratch directory simply as a precaution. When moving
reproduction material out of the current tree, preserve a named immutable
snapshot and document how to restore and verify it.
