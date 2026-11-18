import { describe, expect, it } from 'vitest';
import {
  applySubpixelJitter,
  FrameBudgetGovernor,
  getHaltonJitter,
  halton,
  mat4Identity,
  mat4LookAt,
  mat4Multiply,
  mat4Perspective,
} from '../dlss5Math';

describe('dlss5Math', () => {
  it('computes centered Halton (2,3) jitter within [-0.5, 0.5]', () => {
    expect(halton(1, 2)).toBeCloseTo(0.5, 6);
    expect(halton(2, 2)).toBeCloseTo(0.25, 6);
    for (let i = 0; i < 32; i++) {
      const [jx, jy] = getHaltonJitter(i);
      expect(jx).toBeGreaterThanOrEqual(-0.5);
      expect(jx).toBeLessThanOrEqual(0.5);
      expect(jy).toBeGreaterThanOrEqual(-0.5);
      expect(jy).toBeLessThanOrEqual(0.5);
    }
  });

  it('applies clip-space jitter proportional to inverse render resolution', () => {
    const proj = mat4Perspective(Math.PI / 3, 16 / 9, 0.1, 100);
    const jProj = applySubpixelJitter(proj, 0.25, -0.25, 640, 360);
    expect(jProj[8] - proj[8]).toBeCloseTo(0.5 / 640, 6);
    expect(jProj[9] - proj[9]).toBeCloseTo(-0.5 / 360, 6);
  });

  it('multiplies identity and lookAt matrices cleanly', () => {
    const id = mat4Identity();
    const view = mat4LookAt([0, 2, 5], [0, 1, 0], [0, 1, 0]);
    const prod = mat4Multiply(id, view);
    for (let i = 0; i < 16; i++) {
      expect(prod[i]).toBeCloseTo(view[i], 6);
    }
  });

  it('steps down FrameBudgetGovernor tiers on sustained >7.5ms frames and recovers on <5.2ms', () => {
    const gov = new FrameBudgetGovernor(7.5, 5.2, 3, 5);
    expect(gov.tier).toBe(1);
    gov.onFrameTime(8.5);
    gov.onFrameTime(8.5);
    expect(gov.tier).toBe(1);
    gov.onFrameTime(8.5);
    expect(gov.tier).toBe(2);
    gov.onFrameTime(9.0);
    gov.onFrameTime(9.0);
    gov.onFrameTime(9.0);
    expect(gov.tier).toBe(3);

    for (let i = 0; i < 4; i++) gov.onFrameTime(4.0);
    expect(gov.tier).toBe(3);
    gov.onFrameTime(4.0);
    expect(gov.tier).toBe(2);
    for (let i = 0; i < 5; i++) gov.onFrameTime(4.0);
    expect(gov.tier).toBe(1);
  });
});
