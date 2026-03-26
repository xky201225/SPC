# Synthetic medical SPC-style benchmark (v1)

This package was derived from four MedS3 evaluation files:
- MedQA.json
- MedMCQA.json
- pubmedqa.json
- ddxplus.json

Included files:
- `process_bench` style files: one positive and one negative step-level sample per source problem.
- `delta_bench` style files: same content as process-style, but with DeltaBench field names.
- `prm` style files: one paired correct/incorrect final-step sample per source problem.

Counts:
- MedQA: 1273 source items
- MedMCQA: 4183 source items
- pubmedqa: 500 source items
- ddxplus: 2000 source items
- Total source items: 7956
- Total process-style records: 15912
- Total delta-style records: 15912
- Total prm-style records: 7956
