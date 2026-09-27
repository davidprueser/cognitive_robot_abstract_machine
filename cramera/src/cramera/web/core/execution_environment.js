// The coraplex execution choices of a generated demo: the collision-enabled environment a
// plan performs in, and whether a RobotDemonstration drives the real robot or a
// simulated one. Scripts use the environment name; RobotDemonstration uses its matching
// flag and the ExecutionType member of the chosen name, written into its main() as it is.
(function () {
  'use strict';

  const ENVIRONMENTS = [
    {
      name: 'simulated_robot_advanced',
      label: 'always on — avoid collisions during motion',
      collisionAvoidance: true,
    },
  ];

  window.ExecutionEnvironments = {
    // every environment to offer, in the order they are offered
    all: function () { return ENVIRONMENTS.slice(); },
    // Legacy or missing selections also retain collision avoidance.
    byName: function (name) {
      const found = ENVIRONMENTS.filter(function (e) { return e.name === name; });
      return found.length ? found[0] : ENVIRONMENTS[0];
    },
  };

  const TYPES = [
    {
      name: 'SIMULATED',
      label: 'simulated robot — a world built from the descriptions',
      drivesARealRobot: false,
    },
    {
      name: 'REAL',
      label: 'real robot — the world its world server serves',
      drivesARealRobot: true,
    },
  ];

  window.ExecutionTypes = {
    // every type to offer, in the order they are offered
    all: function () { return TYPES.slice(); },
    // the type of that name, falling back to the simulated robot
    byName: function (name) {
      const found = TYPES.filter(function (type) { return type.name === name; });
      return found.length ? found[0] : TYPES[0];
    },
  };
})();
