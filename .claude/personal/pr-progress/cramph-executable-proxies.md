### Plan
- Ticking side only: owning_process on leaves, ProxyNode, NodeOwnership extension, abstract connection (done in ea41b8fa5e)
- Owner side is the next item, cramph-owned-node-runner (stacked on this branch)

### Done
- Item split (user confirmed); tracking issue #43 comment posted
- Draft PR #46: owning_process field (plain str; composites refuse it), EndBehaviour in owned node JSON, document rewrite into ProxyNode before from_json, run numbers, latched observations, docs section
- Tests: test_proxy_nodes.py 16 pass; test/cramph_test 911 passed / 2 skipped at ea41b8fa5e
- Labels: api-break set (ORM schema gains owning_process column + new tables); new-feature missing on fork

### Next
- Discuss the ORM schema change with cramph code owners before review (api-break)
- Local giskardpy/coraplex QP tests fail on a piqp version mismatch (eps_primal_inf), not this change; CI decides
- Start cramph-owned-node-runner on top of this branch
