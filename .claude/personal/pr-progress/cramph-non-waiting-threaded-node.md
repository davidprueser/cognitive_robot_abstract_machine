## Plan
- BackgroundComputation (non-waiting threaded node) + ThreadedNode start count, shared outcome observation
- Tests first in test/cramph_test/test_statechart/test_background_computation.py

## Done
- Tests written and shown failing; restart test proven against the old overwrite
- Implemented in cramph/src/cramph/threaded_nodes.py; cramph_test passes (415, plus 58 ROS-dependent with ROS sourced)
- krrood format/check clean; committed 9d89822652 and pushed; PR #44 description updated

## Next
- Review by the cramph authors (name BackgroundComputation open for review)
- Then cas-context-and-annotator-node builds ThreadedAnnotatorNode on it
