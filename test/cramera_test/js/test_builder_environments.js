// Unit tests for the environments of web/core/builder_state.js (node:test).
// The Plan Builder lists the world files and the class-built maps in one select, so the
// two kinds have to be told apart again from the one value the select carries: a map
// taken for a file is a path that no parser reads, and a file taken for a map is an
// import of nothing.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

// the state's environment lookup in the form the tests below read it: offered(catalog)
// fills a fresh state and hands back its list, the rest reads that state
function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/builder_state.js'), 'utf8'))();
  const State = global.window.PlanBuilderState;
  let state = new State([]);
  return {
    offered: function (catalog) { state = new State([]); state.offerEnvironments(catalog); return state.offeredEnvironments; },
    value: State.environmentValue,
    byValue: function (offered, value) { return state.environment(value); },
    isMap: State.isMap,
  };
}

const CATALOG = {
  environments: [{name: 'apartment', kind: 'file', path: '/worlds/apartment.urdf'}],
  maps: [{name: 'real-lab apartment', kind: 'map', cls: 'ApartmentEnvironment',
    import: 'from semantic_digital_twin.predetermined_maps.apartment_environment import ApartmentEnvironment'}],
};

test('the files are offered first, then the maps', function () {
  const offered = load().offered(CATALOG);

  assert.deepStrictEqual(offered.map(function (e) { return [e.kind, e.name]; }),
    [['file', 'apartment'], ['map', 'real-lab apartment']]);
});

test('a file is valued by its path and a map by its class, apart from every path', function () {
  const environments = load();
  const [file, map] = environments.offered(CATALOG);

  assert.strictEqual(environments.value(file), '/worlds/apartment.urdf');
  assert.strictEqual(environments.value(map), 'map:ApartmentEnvironment');
});

test('a value names the offered environment it was made from', function () {
  const environments = load();
  const offered = environments.offered(CATALOG);

  assert.strictEqual(environments.byValue(offered, 'map:ApartmentEnvironment'), offered[1]);
  assert.strictEqual(environments.byValue(offered, '/worlds/apartment.urdf'), offered[0]);
});

test('a path the catalog did not list is a file, named after where it lies', function () {
  const environments = load();

  const opened = environments.byValue(environments.offered(CATALOG), '/lab/scan/world.usda');

  assert.deepStrictEqual(opened, {kind: 'file', path: '/lab/scan/world.usda', name: 'scan/world.usda'});
});

test('a map that is not offered names nothing', function () {
  const environments = load();

  assert.strictEqual(environments.byValue(environments.offered(CATALOG), 'map:Nowhere'), null);
  assert.strictEqual(environments.byValue(environments.offered(CATALOG), ''), null);
});

test('only a map is a map', function () {
  const environments = load();
  const [file, map] = environments.offered(CATALOG);

  assert.strictEqual(environments.isMap(map), true);
  assert.strictEqual(environments.isMap(file), false);
  assert.strictEqual(environments.isMap(null), false);
});

test('a map carries what the generated demo imports it with', function () {
  const [, map] = load().offered(CATALOG);

  assert.strictEqual(map.cls, 'ApartmentEnvironment');
  assert.strictEqual(map.import, CATALOG.maps[0].import);
});

test('a catalog without maps offers its files alone', function () {
  const offered = load().offered({environments: CATALOG.environments});

  assert.strictEqual(offered.length, 1);
});
