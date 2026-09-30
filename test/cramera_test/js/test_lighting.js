// Unit tests for core/lighting.js (node:test): the lit/unlit preference, and the
// switch that swaps every mesh's material and restores it untouched.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function loadLighting() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/lighting.js'), 'utf8'))(scope);
  return scope.Lighting;
}

function makeStorage(initial) {
  const items = Object.assign({}, initial);
  return {
    getItem(key) { return key in items ? items[key] : null; },
    setItem(key, value) { items[key] = String(value); },
  };
}

// %% a scene of stand-in meshes
function material(name) {
  return { name: name, disposed: false, dispose() { this.disposed = true; } };
}

function mesh(materialOf) {
  const node = { isMesh: true, material: materialOf, children: [] };
  node.traverse = function (visit) { visit(node); node.children.forEach(function (c) { c.traverse(visit); }); };
  return node;
}

function group(children) {
  const node = { isMesh: false, children: children };
  node.traverse = function (visit) { visit(node); children.forEach(function (c) { c.traverse(visit); }); };
  return node;
}

function unlitOf(lit) { return material('unlit ' + lit.name); }

// %% the default
test('a viewer that has never touched a switch lights the environment and the robots', function () {
  const Lighting = loadLighting();

  assert.strictEqual(Lighting.ENVIRONMENT.on(makeStorage()), true);
  assert.strictEqual(Lighting.ROBOTS.on(makeStorage()), true);
});

// %% remembering the choice
test('switching the environment lighting off survives a reload and leaves the robots lit', function () {
  const Lighting = loadLighting();
  const storage = makeStorage();

  const stored = Lighting.ENVIRONMENT.set(storage, false);

  assert.strictEqual(stored, false);
  assert.strictEqual(Lighting.ENVIRONMENT.on(storage), false);
  assert.strictEqual(Lighting.ROBOTS.on(storage), true);
  assert.strictEqual(Lighting.ENVIRONMENT.set(storage, true), true);
});

// %% swapping materials
test('turning the lighting off gives every mesh the unlit counterpart of its material', function () {
  const Lighting = loadLighting();
  const wall = mesh(material('wall'));
  const robot = mesh([material('shell'), material('visor')]);
  const lighting = new Lighting.Switch(unlitOf);

  lighting.set(false, [group([wall]), robot]);

  assert.strictEqual(wall.material.name, 'unlit wall');
  assert.deepStrictEqual(robot.material.map(function (m) { return m.name; }), ['unlit shell', 'unlit visor']);
});

test('turning the lighting back on restores the lit material and disposes of the unlit one', function () {
  const Lighting = loadLighting();
  const lit = material('wall');
  const wall = mesh(lit);
  const lighting = new Lighting.Switch(unlitOf);
  lighting.set(false, [wall]);
  const unlit = wall.material;

  lighting.set(true, [wall]);

  assert.strictEqual(wall.material, lit);
  assert.strictEqual(unlit.disposed, true);
  assert.strictEqual(lit.disposed, false);
});

test('a mesh that arrives while the lighting is off is drawn unlit, and once only', function () {
  const Lighting = loadLighting();
  const lighting = new Lighting.Switch(unlitOf);
  lighting.set(false, []);
  const late = mesh(material('door'));

  lighting.applyTo(late);
  lighting.applyTo(late);

  assert.strictEqual(late.material.name, 'unlit door');
});

test('a lit scene leaves a mesh that was never swapped alone', function () {
  const Lighting = loadLighting();
  const lit = material('wall');
  const wall = mesh(lit);

  new Lighting.Switch(unlitOf).applyTo(wall);

  assert.strictEqual(wall.material, lit);
});

// %% only a texture has light in it
function textured(name) { return Object.assign(material(name), { map: { image: name } }); }

test('switched to textured surfaces only, a surface of plain colour stays lit', function () {
  const Lighting = loadLighting();
  const scan = mesh(textured('wall'));
  const floorLit = material('floor');
  const floor = mesh(floorLit);
  const lighting = new Lighting.Switch(Lighting.texturedOnly(unlitOf));

  lighting.set(false, [group([scan, floor])]);

  assert.strictEqual(scan.material.name, 'unlit wall');
  assert.strictEqual(floor.material, floorLit);
});

test('turning the lighting back on disposes of no material that stayed lit', function () {
  const Lighting = loadLighting();
  const paint = material('paint');
  const photo = textured('photo');
  const cabinet = mesh([paint, photo]);
  const lighting = new Lighting.Switch(Lighting.texturedOnly(unlitOf));
  lighting.set(false, [cabinet]);
  const [keptLit, unlitPhoto] = cabinet.material;

  lighting.set(true, [cabinet]);

  assert.strictEqual(keptLit, paint);
  assert.deepStrictEqual(cabinet.material, [paint, photo]);
  assert.strictEqual(paint.disposed, false);
  assert.strictEqual(photo.disposed, false);
  assert.strictEqual(unlitPhoto.disposed, true);
});
