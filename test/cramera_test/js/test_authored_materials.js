'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
const PANEL = fs.readFileSync(path.join(WEB, 'panels/robot_scene/panel.js'), 'utf8');

// %% actual panel functions with a deferred mesh loader
class DeferredScene {
  constructor(revision = '129') {
    const pending = this.pending = [];
    const failures = this.failures = [];
    const three = Object.assign({}, THREE, {
      REVISION: revision,
      GLTFLoader: class {
        load(url, loaded, progress, failed) { pending.push(loaded); failures.push(failed); }
      },
    });
    this.context = vm.createContext({
      THREE: three, window: {}, scene3: new THREE.Group(), worldRoot: new THREE.Group(),
      objectMeshes: {}, objectPending: {}, objectLabels: {}, objectIdByKey: {}, objectKeyById: {},
      guidedKeys: new Set(), labelsOn: false, needsRender: false, traj: null, playhead: 0,
      pickDeltas: {}, objOffsetAt: () => ({x: 0, y: 0}), restingBeforePick: () => false,
      makeLabel: () => new THREE.Group(), refreshFrameAxes() {}, arrowOver() {},
      setPose(object, start, end, fraction) {
        object.position.fromArray(start).lerp(new THREE.Vector3().fromArray(end), fraction);
        object.quaternion.fromArray(start.slice(3)).slerp(new THREE.Quaternion().fromArray(end.slice(3)), fraction);
      },
      WOOD_COUNTER: {}, WOOD_TABLE: {}, SCENE: null,
      environmentLighting: {on: true, applyTo() {}, apply() {}},
      lightingOf: () => ({applyTo() {}}),
    });
    this.context.window.EnvironmentTheme = {lookOf: () => ({color: 0xc0c0c0, texture: null, roughness: 0.8, metalness: 0.1})};
    const materialSource = path.join(WEB, 'core/authored-materials.js');
    if (fs.existsSync(materialSource)) {
      vm.runInContext(fs.readFileSync(materialSource, 'utf8'), this.context);
      this.context.AuthoredMaterials = this.context.window.AuthoredMaterials;
    }
    vm.runInContext(fs.readFileSync(path.join(WEB, 'core/shape-specs.js'), 'utf8'), this.context);
    this.context.ShapeSpecs = this.context.window.ShapeSpecs;
    this.install('  function addObject(spec)', '  function removeObject(key)');
    this.install('  function tameMat(mat)', '  function dropGroundToScene()');
    if (PANEL.includes('  function applyObjectFrame(name, frame)')) {
      this.install('  function applyObjectFrame(name, frame)', '  function applyFrame(f)');
    }
    const highlightStart = PANEL.indexOf('  function highlightObjects(ids)');
    const highlightEnd = PANEL.indexOf('    // direct link references:', highlightStart);
    vm.runInContext(PANEL.slice(highlightStart, highlightEnd) + '\n}', this.context);
  }

  install(start, end) {
    vm.runInContext(PANEL.slice(PANEL.indexOf(start), PANEL.indexOf(end, PANEL.indexOf(start))), this.context);
  }

  finish(material) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.01, 0.1), material);
    this.pending.shift()({scene: mesh});
    return mesh;
  }
}

function glass() {
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.04, transmission: 0.98,
    metalness: 0, envMapIntensity: 1.1, emissive: 0x102030, emissiveIntensity: 0.7,
  });
}

// %% authored models and legacy defaults
test('authored model glass retains physical values and avoids opaque shadow maps', () => {
  const scene = new DeferredScene();
  const material = glass();
  const original = material.toJSON();
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
  scene.context.tameModel({obj: mesh, robot: false, preserveMaterials: true});
  assert.deepEqual(material.toJSON(), original);
  assert.equal(mesh.castShadow, false);
  assert.equal(mesh.receiveShadow, true);
  scene.context.tameModel({obj: mesh, robot: false, preserveMaterials: true});
  assert.deepEqual(material.toJSON(), original);
});

test('authored opaque and mixed-material meshes still cast shadows', () => {
  const scene = new DeferredScene();
  for (const material of [new THREE.MeshStandardMaterial(), [glass(), new THREE.MeshStandardMaterial()]]) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
    scene.context.tameModel({obj: mesh, robot: false, preserveMaterials: true});
    assert.equal(mesh.castShadow, true);
    assert.equal(mesh.receiveShadow, true);
  }
});

// %% transmission in the vendored renderer
test('r128 transmission enables blending without changing authored physical values', () => {
  const scene = new DeferredScene('128');
  const material = glass();
  const expected = material.toJSON();
  expected.transparent = true;
  expected.depthWrite = false;
  const transmission = material.transmission;
  scene.context.addObject({id: 'tube', key: 'tube.glb', meshUrl: 'tube.glb', preserveMaterials: true});
  scene.finish(material);
  assert.equal(material.transparent, true);
  assert.equal(material.depthWrite, false);
  assert.equal(material.transmission, transmission);
  assert.deepEqual(material.toJSON(), expected);
});

test('r128 opaque authored materials retain their depth and blending behavior', () => {
  const scene = new DeferredScene('128');
  const material = new THREE.MeshPhysicalMaterial({transmission: 0, roughness: 0.32});
  const expected = material.toJSON();
  scene.context.tameModel({obj: new THREE.Mesh(new THREE.BoxGeometry(), material), preserveMaterials: true});
  assert.deepEqual(material.toJSON(), expected);
});

for (const revision of ['129', '180', null]) {
  test('transmission blending is not forced for renderer revision ' + revision, () => {
    const scene = new DeferredScene(revision);
    const material = glass();
    const expected = material.toJSON();
    scene.context.tameModel({obj: new THREE.Mesh(new THREE.BoxGeometry(), material), preserveMaterials: true});
    assert.deepEqual(material.toJSON(), expected);
  });
}

test('environment models without the explicit opt-in still receive the theme', () => {
  const scene = new DeferredScene();
  for (const preserveMaterials of [undefined, false, 'true']) {
    const material = glass();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material);
    scene.context.tameModel({obj: mesh, robot: false, preserveMaterials});
    const look = scene.context.window.EnvironmentTheme.lookOf('');
    assert.equal(material.color.getHex(), look.color);
    assert.equal(material.roughness, look.roughness);
    assert.equal(material.metalness, look.metalness);
    assert.equal(material.emissive.getHex(), 0);
    assert.equal(mesh.castShadow, true);
  }
});

// %% delayed mesh placement and highlighting
test('authored loose GLBs preserve emission through load and highlight cycles', () => {
  const scene = new DeferredScene();
  const material = glass();
  const original = material.toJSON();
  scene.context.addObject({id: 'tube', key: 'tube.glb', meshUrl: 'tube.glb', preserveMaterials: true});
  const mesh = scene.finish(material);
  assert.deepEqual(material.toJSON(), original);
  assert.equal(mesh.castShadow, false);
  scene.context.highlightObjects([]);
  assert.deepEqual(material.toJSON(), original);
  scene.context.highlightObjects(['tube']);
  assert.notEqual(material.emissive.getHex(), original.emissive);
  scene.context.highlightObjects(['tube']);
  scene.context.highlightObjects([]);
  assert.deepEqual(material.toJSON(), original);
});

test('legacy loose GLBs keep the existing emission suppression', () => {
  const scene = new DeferredScene();
  scene.context.addObject({id: 'tube', key: 'tube.glb', meshUrl: 'tube.glb'});
  const mesh = scene.finish(glass());
  assert.equal(mesh.material.emissive.getHex(), 0);
  assert.equal(mesh.castShadow, true);
});

// %% live object catalog and streamed shape materials
function streamedGlass(preserveMaterials) {
  return {
    id: 'tube', key: 'tube.glb', preserveMaterials,
    shapes: [{kind: 'mesh', mesh: '/mesh?tube', format: 'glb'}],
  };
}

test('live catalog forwards only an explicit authored-material opt-in', async () => {
  const scene = new DeferredScene();
  const policies = [true, false, undefined, 'true'];
  const specs = [];
  Object.assign(scene.context, {
    liveOn: true, liveSyncing: false, liveSpawned: {}, liveStateKeys: {},
    liveUrl: () => 'http://localhost:8765',
    fetch: async () => ({json: async () => ({objects: policies.map((policy, index) => ({
      id: 'tube_' + index, key: 'tube_' + index, kind: 'shapes',
      shapes: streamedGlass(policy).shapes, preserveMaterials: policy,
    }))})}),
    addObject: spec => specs.push(spec),
  });
  scene.install('  function syncLiveObjects()', '  // watching a running demo');
  scene.context.syncLiveObjects();
  await new Promise(setImmediate);
  assert.equal(specs.length, policies.length);
  assert.deepEqual(specs.map(spec => spec.preserveMaterials), policies.map(policy => policy === true));
});

test('streamed authored GLBs preserve glass shading and restore emission after highlights', () => {
  const scene = new DeferredScene('128');
  const material = glass();
  const expected = material.toJSON();
  expected.transparent = true;
  expected.depthWrite = false;
  scene.context.addObject(streamedGlass(true));
  const mesh = scene.finish(material);
  assert.deepEqual(material.toJSON(), expected);
  assert.equal(mesh.castShadow, false);
  assert.equal(mesh.receiveShadow, true);
  scene.context.highlightObjects(['tube']);
  assert.notEqual(material.emissive.getHex(), expected.emissive);
  scene.context.highlightObjects([]);
  assert.deepEqual(material.toJSON(), expected);
});

test('streamed GLBs without opt-in keep their material values and receive standard shadows', () => {
  const scene = new DeferredScene('128');
  const material = glass();
  const expected = material.toJSON();
  scene.context.addObject(streamedGlass(false));
  const mesh = scene.finish(material);
  assert.deepEqual(material.toJSON(), expected);
  assert.equal(mesh.castShadow, true);
  assert.equal(mesh.receiveShadow, true);
  assert.equal(mesh.userData.preserveMaterials, undefined);
});

for (const preserveMaterials of [true, false]) {
  test('failed streamed meshes prepare fallback materials with preservation ' + preserveMaterials, () => {
    const scene = new DeferredScene('128');
    scene.context.addObject(streamedGlass(preserveMaterials));
    scene.failures.shift()();
    const meshes = [];
    scene.context.objectMeshes['tube.glb'].traverse(object => { if (object.isMesh) meshes.push(object); });
    assert.equal(meshes.length, 1);
    assert.equal(meshes[0].castShadow, true);
    assert.equal(meshes[0].receiveShadow, true);
    assert.equal(meshes[0].userData.preserveMaterials, preserveMaterials ? true : undefined);
  });
}

test('scene descriptions forward the material policy of models and loose objects', () => {
  const scene = new DeferredScene();
  const specs = [];
  Object.assign(scene.context, {
    sceneBase: '/scenes/lab/', models: [], statusEl: null, manager: {},
    makeUrdfLoader: () => ({load(url, callback) { callback(new THREE.Group()); }}),
    ModelPoses: {primary() {}}, refreshJointControls() {}, buildPlaceTargetMarker() {}, setTimeout() {},
    addObject: spec => specs.push(spec),
  });
  scene.install('  function loadScene(sc)', '  function finalize()');
  scene.context.loadScene({
    name: 'lab', models: [{name: 'bench', urdf: 'bench.urdf', preserveMaterials: true}],
    objects: [{id: 'tube', key: 'tube.glb', mesh: 'tube.glb', preserveMaterials: true}],
  });
  assert.equal(scene.context.models[0].preserveMaterials, true);
  assert.equal(specs[0].preserveMaterials, true);
});

test('the authored-material helper loads before the scene panel', () => {
  const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
  const helper = html.indexOf('src="core/authored-materials.js"');
  assert.ok(helper >= 0);
  assert.ok(helper < html.indexOf('src="panels/robot_scene/panel.js"'));
});
