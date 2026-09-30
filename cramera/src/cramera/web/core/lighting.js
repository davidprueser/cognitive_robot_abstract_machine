/* ============================================================================
 * core/lighting.js — whether a part of the scene is lit, or drawn as its colours and
 * textures stand.
 *
 * Lit, a surface takes the viewer's lamps, shadows and reflections. Unlit, each mesh
 * is drawn with a flat counterpart of its material, which is how a photographed scan
 * reads best: its light is already in the pixels. The environment and the robots are
 * switched separately, since a scan reads best unlit while a robot's modelled shell
 * needs light to have any shape at all. Each preference survives a reload; the swap
 * keeps every lit material so switching back restores it untouched.
 *
 * Only a texture has light in it. A surface of plain colour - a floor laid in as a box
 * where the scan was cleaned up - has none, and drawn flat it turns the colour it was
 * authored as, which is near white; so the environment's switch leaves it lit.
 *
 * No DOM and no three.js of its own: the panel hands in how a lit material becomes an
 * unlit one, so the switch is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  // %% the remembered choice
  class Preference {
    /** @param {string} key localStorage key of whether this part of the scene is lit. */
    constructor(key) {
      this.key = key;
    }

    /** Whether the part is lit; a viewer that never touched the switch lights it. */
    on(storage) {
      return storage.getItem(this.key) !== 'false';
    }

    /** Switch the lighting on or off; returns the state as stored. */
    set(storage, lit) {
      storage.setItem(this.key, lit ? 'true' : 'false');
      return this.on(storage);
    }
  }

  const ENVIRONMENT = new Preference('cramera.lighting.environment');
  /* Whether the environment - the building, its furniture and loose objects - is lit. */

  const ROBOTS = new Preference('cramera.lighting.robots');
  /* Whether the robots are lit. */

  // %% which surfaces go unlit
  /**
   * An unlit counterpart for textured surfaces only, whose light is in their pixels;
   * any other surface is handed back as it is, and so stays lit.
   * @param {function} unlitOf Turns one lit material into its unlit counterpart.
   */
  function texturedOnly(unlitOf) {
    return function (mat) { return mat && mat.map ? unlitOf(mat) : mat; };
  }

  // %% swapping every mesh's material
  class Switch {
    /**
     * @param {function} unlitOf Turns one lit material into its unlit counterpart, or
     *     hands it back unchanged for one that stays lit.
     */
    constructor(unlitOf) {
      this.unlitOf = unlitOf;
      this.lit = new WeakMap();   // mesh -> the material(s) it is lit with
      this.on = true;
    }

    /** Draw one mesh the way the switch stands, swapping its material if it must. */
    applyTo(mesh) {
      if (this.on) {
        if (!this.lit.has(mesh)) return;
        const unlit = mesh.material;
        const lit = this.lit.get(mesh);
        const kept = new Set(Array.isArray(lit) ? lit : [lit]);
        mesh.material = lit;
        this.lit.delete(mesh);
        (Array.isArray(unlit) ? unlit : [unlit]).forEach(function (m) {
          if (!kept.has(m) && m.dispose) m.dispose();
        });
        return;
      }
      if (this.lit.has(mesh)) return;
      this.lit.set(mesh, mesh.material);
      mesh.material = Array.isArray(mesh.material) ? mesh.material.map(this.unlitOf) : this.unlitOf(mesh.material);
    }

    /** Draw every mesh below a root the way the switch stands. */
    apply(root) {
      const self = this;
      root.traverse(function (c) { if (c.isMesh) self.applyTo(c); });
    }

    /** Turn the lighting on or off for everything below the roots given. */
    set(lit, roots) {
      this.on = lit;
      roots.forEach(this.apply, this);
    }
  }

  global.Lighting = {
    Preference: Preference, ENVIRONMENT: ENVIRONMENT, ROBOTS: ROBOTS,
    Switch: Switch, texturedOnly: texturedOnly,
  };
})(window);
