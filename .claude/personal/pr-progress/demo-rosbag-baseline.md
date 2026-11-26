## PR #45 demo-rosbag-baseline (fork, draft) - worktree ~/repos/cognitive_robot_abstract_machine_demo_rosbag_baseline

Plan
- dataset/breakfast_table_rosbag.py: bag reader, frame publisher, expected demo outcomes (frames 0, 27, 54)
- test_demo_rosbag_baseline.py: run demo.py's pipeline once per frame (plane seeded 0), assert plane, inliers, hypothesis count, world bodies
- format with ruff + docformatter, krrood check

Done
- Probed: seeded demo is deterministic; world holds only the camera (kinect_wo_tf has no pose)
- Branch, draft PR #45, plan item in progress

Next
- Write dataset module and test, run locally, push
