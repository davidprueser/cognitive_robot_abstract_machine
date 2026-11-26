### Plan
- `owner` field on StatechartNode (leaves only; composites refuse an owner); owned nodes write their end behaviour into their JSON
- `ProxyNode`: end behaviour from the JSON, writes back the JSON it came from, run number per start, latched observation per run
- `NodeOwnership` context extension (own name + abstract connection); `Statechart` deserialization makes proxies for leaves owned elsewhere
- Tests first in test/cramph_test/test_statechart/test_proxy_nodes.py with a recording connection

### Done
- Item split (user confirmed): owner side is new item cramph-owned-node-runner; tracking issue #43 comment posted
- Branch + draft PR #46 (fork); label `new-feature` missing on fork

### Next
- Write failing tests, then implement in a new cramph/src/cramph/ownership.py
