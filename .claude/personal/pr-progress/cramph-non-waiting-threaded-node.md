## Plan
- Write failing tests in test/cramph_test/test_statechart/test_threaded_nodes.py (tick does not wait, failures, superseded run discarded)
- ThreadedNode: start count, failure_types, shared outcome-to-observation method
- FunctionCall uses the shared method; new BackgroundComputation observes UNKNOWN while running
- ruff + docformatter + krrood check via pre-commit; run cramph threaded and function-call tests

## Done
- Branch and draft PR #44 opened from main

## Next
- Failing tests first
