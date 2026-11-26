# RoboKudo on cramph: standing rules

- Port incrementally as a stack of pull requests; every PR leaves today's py_trees engines working
  until the migration wave removes them.
- A ported annotator keeps one computational core, used by its cramph node and by a thin py_trees
  shell until the shell is deleted; never copy annotator logic.
- Annotator nodes are dataclasses configured by their own fields, built on main, independent of
  fork PR #18 and cram2#933.
- The CAS never leaves RoboKudo's process: it lives in a context extension that the pipeline's
  executor adds and the pipeline node refills on every start. One pipeline per statechart.
- World writes happen on the tick thread, never in a node's worker thread (CasADi is not
  thread-safe).
- A ticking process never waits inside its tick for a thread or another process; remote and
  threaded outcomes are latched values it reads.
- Generic statechart behaviour (threaded nodes, ownership, proxies, transport) goes into cramph,
  with ROS only as a soft dependency; perception goes into robokudo.
- Rosbag tests run locally only and are skipped when the bag is absent; CI covers the same nodes
  with the stored CAS frames in robokudo_test_data.
- No cross-framework communication before the shared-statechart wave; the proof of concept is
  standalone RoboKudo.
- The cramph authors' design (RoboKudo on cramph doc, 2026-11-25) and the original proposal
  artifact are orientation, not decisions; a decision lands in an item's notes.
