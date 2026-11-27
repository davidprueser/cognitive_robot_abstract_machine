## Plan
- FunctionCall observes UNKNOWN instead of waiting (cramph maintainer's request); ThreadedNode start count kept
- BackgroundComputation removed; tests merged into test_function_call.py

## Done
- 80d747f1b6 pushed: FunctionCall non-waiting, BackgroundComputation gone; cramph_test 897 passed with ROS; krrood format/check clean
- PR #44 title, description and labels (api-break, bug-fix) updated

## Next
- Check CI, especially the coraplex FunctionCall tests (coraplex ORM does not generate locally)
- Finalize with the cramph maintainer; the user merges it, then the PoC items start
- Open: remove the now unused ThreadedNode.wait_until_finished?
