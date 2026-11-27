## Plan
- robokudo.statechart: CASAccess, CASNode, AnnotatorNode, ThreadedAnnotatorNode, PipelineNode, SecondPipelineError
- robokudo depends on cramph (user approved)

## Done
- Implemented with 16 tests in test/robokudo_test/test_statechart (pass, stable over 5 runs); krrood format/check clean
- Committed 56372f81ea and pushed; draft PR #47 description matches

## Next
- User review of PR #47 (decisions: annotate returns bool, CASNode for utility nodes, one pipeline per context)
- robokudo-perception-executor adds CASAccess to its default context extensions
