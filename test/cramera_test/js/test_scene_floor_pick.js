// The scene panel's floor pick: the plan builder arms the view for one click, and the
// view answers with the floor point under the cursor in the map frame.
'use strict';

const assert = require('node:assert/strict');

/** Messages cross a vm realm, so their prototypes differ: compare them as plain data. */
function plain(value) { return JSON.parse(JSON.stringify(value)); }
const path = require('node:path');
const test = require('node:test');

const ScenePanelFunctions = require('./scene_panel_functions');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% the panel's pointer functions against a camera looking straight down
function panelLookingDownAt(x, z) {
  const sent = [];
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(x, 5, z);
  camera.up.set(0, 0, -1);           // looking straight down, the world's up cannot be the camera's
  camera.lookAt(x, 0, z);
  camera.updateMatrixWorld();
  const worldRoot = new THREE.Group();
  worldRoot.rotation.x = -Math.PI / 2;
  worldRoot.updateMatrixWorld();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1));
  ground.position.set(0, 0.3, 0);
  const domElement = {
    style: {}, captured: [], getBoundingClientRect: () => ({left: 0, top: 0, width: 200, height: 200}),
    setPointerCapture(id) { this.captured.push(id); },
  };
  const scope = new ScenePanelFunctions({
    THREE, camera, worldRoot, ground, renderer: {domElement},
    dragNdc: new THREE.Vector2(), _ray: new THREE.Raycaster(), _hitPt: new THREE.Vector3(),
    _floorPlane: new THREE.Plane(), UP: new THREE.Vector3(0, 1, 0), round3: (v) => Math.round(v * 1000) / 1000,
    floorPickArmed: false, floorPickAnchor: null, floorPickArrow: null, needsRender: false, HEADING_DRAG_MINIMUM: 0.02,
    controls: {enabled: true}, partClickCb: () => assert.fail('a floor pick is not a click on a part'),
    window: {parent: {postMessage(message) { sent.push(message); }}},
  }).scope;
  scope.window.parent.postMessage = (message) => sent.push(message);
  scope.sent = sent;
  scope.domElement = domElement;
  return scope;
}

const CENTER = {clientX: 100, clientY: 100};

// %% arming
test('the builder arms the view for a floor pick and the cursor says so', () => {
  const scope = panelLookingDownAt(0, 0);
  scope.handleParentMessage({type: 'cramera-pick-floor', on: true});
  assert.equal(scope.floorPickArmed, true);
  assert.equal(scope.domElement.style.cursor, 'crosshair');
  scope.handleParentMessage({type: 'cramera-pick-floor', on: false});
  assert.equal(scope.floorPickArmed, false);
  assert.equal(scope.domElement.style.cursor, '');
});

// %% the point under the cursor
test('the floor point under the cursor is given in the map frame', () => {
  const scope = panelLookingDownAt(1, 2);
  const point = scope.floorPointAt(CENTER);
  // the world is y-up and the map z-up: world (1, floor, 2) is map (1, -2, floor)
  assert.ok(Math.abs(point.x - 1) < 1e-6, String(point.x));
  assert.ok(Math.abs(point.y + 2) < 1e-6, String(point.y));
  assert.ok(Math.abs(point.z - 0.3) < 1e-6, String(point.z));
});

// %% press, drag, release
const ASIDE = {clientX: 150, clientY: 100, pointerId: 7};

test('a press and release in one place answers with the floor point and no heading', () => {
  const scope = panelLookingDownAt(1, 2);
  scope.handleParentMessage({type: 'cramera-pick-floor', on: true});
  scope.beginFloorPick(Object.assign({pointerId: 7}, CENTER));
  assert.equal(scope.controls.enabled, false, 'the camera stays put while the heading is dragged');
  assert.deepEqual(scope.domElement.captured, [7]);
  scope.finishFloorPick(CENTER);
  assert.deepEqual(plain(scope.sent), [{type: 'cramera-floor-picked', x: 1, y: -2}]);
  assert.equal(scope.floorPickArmed, false);
  assert.equal(scope.controls.enabled, true);
  assert.equal(scope.domElement.style.cursor, '');
});

test('dragging from the pressed point turns the heading the way the arrow points', () => {
  const scope = panelLookingDownAt(1, 2);
  scope.handleParentMessage({type: 'cramera-pick-floor', on: true});
  const anchor = scope.floorPointAt(CENTER);
  const tip = scope.floorPointAt(ASIDE);
  const heading = Math.atan2(tip.y - anchor.y, tip.x - anchor.x);
  scope.beginFloorPick(Object.assign({pointerId: 7}, CENTER));
  scope.dragFloorPick(ASIDE);
  assert.equal(scope.floorPickArrow.visible, true);
  assert.ok(Math.abs(scope.floorPickArrow.rotation.z - heading) < 1e-9);
  assert.ok(Math.abs(scope.floorPickArrow.position.x - anchor.x) < 1e-9, 'the arrow stands where the press was');
  scope.finishFloorPick(ASIDE);
  const [answer] = plain(scope.sent);
  assert.equal(answer.type, 'cramera-floor-picked');
  assert.equal(answer.x, 1);
  assert.equal(answer.y, -2);
  assert.ok(Math.abs(answer.yaw - heading) < 1e-3, `${answer.yaw} vs ${heading}`);
  assert.equal(scope.floorPickArrow.visible, false);
});

test('a press while not armed begins no floor pick', () => {
  const scope = panelLookingDownAt(1, 2);
  scope.beginFloorPick(Object.assign({pointerId: 7}, CENTER));
  assert.equal(scope.floorPickAnchor, null);
  assert.equal(scope.controls.enabled, true);
});

test('a click while not armed is not answered as a floor pick', () => {
  const scope = panelLookingDownAt(1, 2);
  scope.partClickCb = null;
  scope.classifyClick = () => null;
  scope.finishClick(CENTER);
  assert.deepEqual(plain(scope.sent), []);
});
