// Unit tests for the execution types of web/core/execution_environment.js (node:test).
// The Plan Builder offers these as its "robot" choice, and the generated
// RobotDemonstration is handed the member of that name, so a lookup that loses the
// name or its meaning produces a demo driving the wrong robot.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/execution_environment.js'), 'utf8'))();
  return global.window.ExecutionTypes;
}

test('a simulated and a real robot are offered, the simulated one first', function () {
  assert.deepStrictEqual(
    load().all().map(function (type) { return [type.name, type.drivesARealRobot]; }),
    [['SIMULATED', false], ['REAL', true]],
  );
});

test('every offered type carries a label to show', function () {
  load().all().forEach(function (type) {
    assert.ok(type.label, type.name);
  });
});

test('byName finds the type of that name', function () {
  const types = load();

  assert.strictEqual(types.byName('REAL').drivesARealRobot, true);
  assert.strictEqual(types.byName('SIMULATED').drivesARealRobot, false);
});

test('an unknown or missing name falls back to the simulated robot', function () {
  const types = load();

  assert.deepStrictEqual(types.byName(''), types.all()[0]);
  assert.deepStrictEqual(types.byName(null), types.all()[0]);
  assert.strictEqual(types.byName('SEMI_REAL').drivesARealRobot, false);
});

test('the offered types are not the module\'s own list', function () {
  const types = load();

  types.all().pop();

  assert.strictEqual(types.all().length, 2);
});
