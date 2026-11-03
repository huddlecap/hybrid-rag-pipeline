Build Phase 3, Part B: Python Verification & Fallback Logic.

- Update `src/generator.py` to add a validation layer over the LLM's JSON output.
- Write a regex-based verification function that scans the 'answer' string for all `[x]` citations.
- Verify that every `x` is a valid integer between 1 and the length of the provided chunks list. If an invalid citation (e.g., `[6]`) is found, remove it from the string or flag it.
- Add explicit fallback logic: If the 'answer' string is exactly "I don't know" (case-insensitive), override the 'confidence' score in the JSON to 0.0, regardless of what the LLM outputted.
- Create a unit test script `scripts/test_verifier.py` to test this logic locally without calling the API. Feed it two mock JSON strings:
  1. The exact JSON from Part A (should pass verification).
  2. A mock string with an out-of-bounds citation `[6]` and a mock string saying "I don't know" with a fake 0.9 confidence (should trigger the fallbacks).

Report, short:

- The regex pattern used to catch the citations.
- The true/false results of the two unit tests.
  Then STOP.
