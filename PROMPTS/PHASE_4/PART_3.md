We are proceeding with the speed-optimized, defensible chunking comparison plan. Execute these setup steps:

1. **Strategy-Specific Golden Datasets:** Generate `golden_dataset_fixed.json`, `golden_dataset_structure.json`, and `golden_dataset_semantic.json` for a 30-question subset. Map the correct chunk IDs (allowing multiple acceptable IDs if an answer spans chunks).
2. **Harness Routing:** Update `scripts/evaluate_pipeline.py` to accept a `--strategy` flag. This flag must simultaneously route the vector DB collection, the specific golden dataset, and a strategy-specific checkpoint file (e.g., `checkpoint_fixed.json`).
3. **Deterministic Single-Pass Judge:** Refactor the Groq judge prompt to ask for this exact JSON schema: `{"correctness": 0, "faithfulness": 0, "unsupported_citations": ["[2]"]}`. Calculate the binary citation accuracy in Python (0 if unsupported citations exist, 1 if empty). Include 1 retry for invalid JSON.
4. **Smart Comparison Script:** Write `scripts/compare_chunking.py` to read the three checkpoint files. It must explicitly display `N` (number of evaluations), mark partial results, calculate paired wins/losses, and refuse to declare an overall winner if `N` is mismatched.

Report short:

- Confirm all 4 setup steps are complete and we are ready to run Cycle 1.
  Then STOP.
