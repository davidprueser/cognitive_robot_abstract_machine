// The plan builder's "Place robot" tool: arm the scene for one click, take the floor
// point it answers with as the active robot's position, and stand the robot there.
'use strict';

const assert = require('node:assert/strict');

/** Messages cross a vm realm, so their prototypes differ: compare them as plain data. */
function plain(value) { return JSON.parse(JSON.stringify(value)); }
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

// %% the authoring page with a scene frame that records what it is told
class PlacementPage {
  constructor() {
    this.elements = new Map();
    this.requests = [];
    this.framed = [];
    this.context = vm.createContext({
      window: {addEventListener() {}, location: {hostname: 'localhost'}},
      document: {getElementById: (id) => this.element(id), createElement: () => this.element('option' + this.elements.size)},
      fetch: async (url, options) => {
        const route = new URL(url, 'http://localhost').pathname;
        this.requests.push({route, body: options && options.body ? JSON.parse(options.body) : null});
        return {ok: true, json: async () => ({ok: true, robots: [], objects: {}})};
      },
      setInterval() { return 1; }, clearInterval() {}, setTimeout() {},
    });
    const web = path.join(__dirname, '../../../cramera/src/cramera/web');
    for (const module of ['base_control', 'execution_environment', 'plan_steps', 'plan_constraints', 'builder_state']) {
      const filename = path.join(web, 'core', module + '.js');
      vm.runInContext(fs.readFileSync(filename, 'utf8'), this.context, {filename});
    }
    this.context.PlanConstraints = this.context.window.PlanConstraints;
    const source = fs.readFileSync(path.join(web, 'plan_builder.js'), 'utf8');
    const exportsSource = fs.readFileSync(path.join(__dirname, '../dataset/builder_multi_robot_exports.js'), 'utf8');
    vm.runInContext(source.slice(0, source.indexOf('  // ---------- boot ----------')) + exportsSource, this.context, {filename: path.join(web, 'plan_builder.js')});
    this.api = this.context.window.multiRobotPage;
    this.api.initialize([{name: 'PR2', cls: 'PR2', import: 'from semantic_digital_twin.robots.pr2 import PR2', arms: ['BOTH'], steps: ['park_arms']}]);
    this.api.state.updateRobot(this.robot.id, {x: 6.5, y: 1, yaw: 0.5});
    this.api.renderRobotInstances();
  }
  element(identifier) {
    if (!this.elements.has(identifier)) this.elements.set(identifier, {
      value: '', textContent: '', innerHTML: '', className: '', style: {}, children: [], attributes: {},
      contentWindow: {postMessage: (message) => this.framed.push(message)},
      addEventListener() {}, setAttribute(name, value) { this.attributes[name] = value; },
      replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); },
    });
    return this.elements.get(identifier);
  }
  get robot() { return this.api.state.activeRobot(); }
  get button() { return this.element('pb-place-robot'); }
  async settle() { await new Promise(setImmediate); }
}

// %% arming the scene
test('the tool arms the scene frame for a floor pick and shows itself pressed', () => {
  const page = new PlacementPage();
  page.api.toggleFloorPlacement();
  assert.deepEqual(plain(page.framed), [{type: 'cramera-pick-floor', on: true}]);
  assert.equal(page.button.attributes['aria-pressed'], 'true');
  page.api.toggleFloorPlacement();
  assert.deepEqual(plain(page.framed[1]), {type: 'cramera-pick-floor', on: false});
  assert.equal(page.button.attributes['aria-pressed'], 'false');
});

// %% taking the picked point
test('the picked floor point becomes the active robot\'s position and stands it in the scene', async () => {
  const page = new PlacementPage();
  page.api.live = true;
  page.api.toggleFloorPlacement();
  page.api.handleSceneMessage({type: 'cramera-floor-picked', x: 2.345, y: -1.5});
  await page.settle();
  assert.equal(page.robot.x, 2.35);
  assert.equal(page.robot.y, -1.5);
  assert.equal(page.robot.yaw, 0.5);
  assert.equal(page.element('pb-rx').value, 2.35);
  assert.equal(page.element('pb-ry').value, -1.5);
  const placed = page.requests.find((request) => request.route === '/robot/place');
  assert.ok(placed, 'the running scene is asked to stand the robot there');
  assert.equal(placed.body.x, 2.35);
  assert.equal(placed.body.y, -1.5);
  assert.equal(placed.body.yaw, 0.5);
  // one click, then the tool is off again
  assert.deepEqual(plain(page.framed[1]), {type: 'cramera-pick-floor', on: false});
  assert.equal(page.button.attributes['aria-pressed'], 'false');
});

test('a floor point arriving while the tool is off leaves the robot where it is', () => {
  const page = new PlacementPage();
  page.api.handleSceneMessage({type: 'cramera-floor-picked', x: 2.345, y: -1.5});
  assert.equal(page.robot.x, 6.5);
  assert.equal(page.robot.y, 1);
  assert.deepEqual(plain(page.framed), []);
});
