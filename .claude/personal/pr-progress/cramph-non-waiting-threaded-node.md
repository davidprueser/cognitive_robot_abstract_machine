## Plan
- FunctionCall observes UNKNOWN instead of waiting (cramph maintainer's request); ThreadedNode start count kept
- BackgroundComputation removed; tests merged into test_function_call.py

## Done
- Squashed into 24124716de (force-pushed): FunctionCall non-waiting, wait_until_finished removed; cramph_test 897 passed; krrood clean
- PR #44 description restructured (one change plus the bug it exposes); labels api-break, bug-fix

## Next
- Open upstream as a draft so CI runs the coraplex FunctionCall tests (coraplex ORM does not generate locally)
- Finalize with the cramph maintainer; the user merges it, then the PoC items start
