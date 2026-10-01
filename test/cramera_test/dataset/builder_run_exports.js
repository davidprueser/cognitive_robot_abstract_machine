  window.builderRunTest = {
    begin(live = true) { liveOn = live; return ++_runMonitor; },
    monitorRun, pollLive, stopRunMonitor, stopLive, startLive, runPlan, reloadScene,
    get live() { return liveOn; },
    generatedSteps: [],
    clearSteps() { steps = []; },
    offerRobot(robot) { builderState = new window.PlanBuilderState([robot]); builderState.addRobot(robot.name, {}); },
    rejectCapture() { synchronizeObjects = function () { return Promise.reject(new Error('capture unavailable')); }; },
    prepare() {
      robotInfo = function () { return {name: 'PR2'}; };
      generate = generateSelected = function (generatedSteps) {
        window.builderRunTest.generatedSteps.push(generatedSteps);
        return 'generated fixture';
      };
      synchronizeObjects = function () { return Promise.resolve(); };
      fetchSurfaces = showModelStatus = toast = function () {};
      steps = [{type: 'park_arms', params: {arm: 'BOTH'}}];
    },
  };
})();
