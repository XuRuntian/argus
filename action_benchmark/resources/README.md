# Bundled vocabulary

`skill_spec_v1.yaml` is copied unchanged from
`robo-subtask-alignment/src/robo_subtask_alignment/vlm_annotation/resources/skill_spec_v1.yaml`,
commit `6214b309c3712f172b02c3bb20a56b3ca31e96fc` (Apache-2.0).
It contains both skill and phase definitions; no external repository is needed at runtime.
The source repository license is retained in `LICENSE.upstream`.

The generic contract hashes the vocabulary and includes its argument definitions,
skill templates and phase rules. Changes therefore produce a different contract.
This is a reusable vocabulary, not evidence that its authoring history was isolated
from every evaluation dataset. Freeze it before evaluating; do not tune it on test answers.
