  builderState = new window.PlanBuilderState(scenario.robots);
  if (scenario.catalog) builderState.offerEnvironments(scenario.catalog);
  if (scenario.instances) {
    scenario.instances.forEach((instance) => builderState.addRobot(instance.model, instance));
    builderState.selectInstance(scenario.activeIdentifier, []);
  }
  if (scenario.environmentJointPositions) builderState.environmentJointPositions = scenario.environmentJointPositions;
  objects = scenario.objects;
  steps = scenario.steps;
  if (scenario.robotXY) robotXY = scenario.robotXY;
  builderState.capture(objects, scenario.captured);
  // a generator refusing the authored combination reports why instead of a demo
  function attempt(generator) { try { return generator(); } catch (error) { return {refused: error.message}; } }
  window.generatedDemos = {script: attempt(generate), class: attempt(generateClass)};
  if (scenario.inspectStart) {
    window.generatedDemos.start = {position: robotXY, inputs: {x: Number($('pb-rx').value), y: Number($('pb-ry').value)}};
  }
})();
