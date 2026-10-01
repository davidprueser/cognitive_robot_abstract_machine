// What the scene panel draws per frame, and what it leaves out: a scanned building
// stays out of the shadow map, the occlusion chain renders the scene once, and a
// scene nothing moves in is not redrawn while its models settle.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const ScenePanelFunctions = require('./scene_panel_functions');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
const PANEL = fs.readFileSync(path.join(WEB, 'panels/robot_scene/panel.js'), 'utf8');

// %% the panel's model functions against a scene without a renderer
class ModelLoader {
  constructor() {}
  defaultMeshLoader() {}
  load(url, callback) { callback(new THREE.Group()); }
}

function panelFunctions(extra = {}) {
  const looked = [];
  const state = Object.assign({
    THREE, URDFLoader: ModelLoader, scene3: new THREE.Group(), worldRoot: new THREE.Group(),
    models: [], objectMeshes: {}, needsRender: false, SCENE: null, sceneBase: '',
    statusEl: null, manager: {}, robotModel: null, playbackSpeedMultiplier: 1, camera: {},
    window: {EnvironmentTheme: {lookOf() {
      looked.push(true);
      return {color: 0xc0c0c0, texture: null, roughness: 0.8, metalness: 0.1};
    }}},
    ModelPoses: {primary() { return null; }}, setPose() {}, refreshFrameAxes() {},
    refreshJointControls() {}, buildPlaceTargetMarker() {}, setTimeout() {},
    addObject() {}, OWN_LIGHTS: new Set(), lightingOf: () => ({applyTo() {}}),
    environmentLighting: {on: true, applyTo() {}}, robotLighting: {applyTo() {}},
    frameDisplay: {visible: false, hidden: new Set()}, frameTriads: [], $: () => null,
    marker: new THREE.Group(), _embeddedScene: true,
  }, extra);
  const scope = new ScenePanelFunctions(state, ['authored-materials.js']).scope;
  scope.AuthoredMaterials = scope.window.AuthoredMaterials;
  scope.looked = looked;
  return scope;
}

function opaqueMesh() {
  return new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial());
}

// %% the shadow map
test('a scene model told not to cast shadows stays out of the shadow map', () => {
  const scope = panelFunctions();
  const mesh = opaqueMesh();
  scope.tameModel({obj: mesh, robot: false, preserveMaterials: true, castsShadows: false});
  assert.equal(mesh.castShadow, false);
  assert.equal(mesh.receiveShadow, true);
});

test('a scene model saying nothing about shadows casts them as before', () => {
  const scope = panelFunctions();
  for (const entry of [{robot: false, preserveMaterials: true}, {robot: false}, {robot: true}]) {
    const mesh = opaqueMesh();
    scope.tameModel(Object.assign({obj: mesh}, entry));
    assert.equal(mesh.castShadow, true, JSON.stringify(entry));
  }
});

test('a scene description says which of its models cast shadows', () => {
  const scope = panelFunctions();
  scope.loadScene({
    name: 'lab',
    models: [
      {name: 'scan', urdf: 'scan.urdf', castsShadows: false},
      {name: 'robot', urdf: 'robot.urdf', robot: true},
    ],
  });
  assert.deepEqual(scope.models.map(model => model.castsShadows), [false, true]);
});

// %% the occlusion chain
test('the occlusion chain renders the scene once, inside the occlusion pass', () => {
  const added = [];
  class Composer { constructor() {} addPass(pass) { added.push(pass); } }
  class RenderPass {}
  class ShaderPass { constructor(shader) { this.shader = shader; } }
  class OcclusionPass { constructor(scene, camera, width, height) { this.size = [width, height]; } }
  const context = vm.createContext({
    THREE: {EffectComposer: Composer, RenderPass, ShaderPass, CopyShader: {}},
    window: {BackgroundIgnoringSSAOPass: OcclusionPass},
    BackgroundIgnoringSSAOPass: OcclusionPass,
    container: {clientWidth: 640, clientHeight: 480}, renderer: {}, scene3: {}, camera: {},
  });
  const start = PANEL.indexOf('  let composer = null, ssaoPass = null;');
  const end = PANEL.indexOf('  })();', start) + '  })();'.length;
  assert.ok(start >= 0 && end > start);
  const passes = vm.runInContext(PANEL.slice(start, end) + '\n({composer, ssaoPass})', context);
  assert.ok(passes.composer instanceof Composer);
  assert.deepEqual(added.map(pass => pass.constructor), [OcclusionPass, ShaderPass]);
  assert.equal(added[0], passes.ssaoPass);
  assert.equal(added[1].renderToScreen, true);
});

// %% the render loop
function idleLoop() {
  const drawn = [];
  const scope = panelFunctions({
    requestAnimationFrame() {}, running: true, clock: new THREE.Clock(), composer: null,
    renderer: {render: () => drawn.push(true)},
    replayClip: null, highlightArrows: {}, playing: false, traj: null, liveOn: false,
    cameraController: {update() { return false; }}, controls: {autoRotate: false},
    stepReplay() {}, playheadCbs: [], follow: false,
  });
  scope.drawn = drawn;
  return scope;
}

test('a scene nothing moves in is not redrawn while its models settle', () => {
  const scope = idleLoop();
  // an untamed environment mesh: re-taming it would ask the theme for its look
  scope.models.push({obj: opaqueMesh(), robot: false});
  scope.tick();
  assert.deepEqual(scope.looked, []);
  assert.deepEqual(scope.drawn, []);
});

test('a scene asked for a redraw is drawn once', () => {
  const scope = idleLoop();
  scope.needsRender = true;
  scope.tick();
  scope.tick();
  assert.equal(scope.drawn.length, 1);
  assert.equal(scope.needsRender, false);
});
