import { describe, expect, it } from 'vitest';
import { clarity, CpuImg, edgeAa, luma, RgbTuple, vibrance } from '../enhanceMath';

function sat(c: RgbTuple): number {
  return Math.max(c[0], c[1], c[2]) - Math.min(c[0], c[1], c[2]);
}

function staircase(): CpuImg {
  const img = new CpuImg(24, 24);
  for (let y = 0; y < 24; y++) {
    for (let x = 0; x < 24; x++) {
      img.gray(x, y, x > y ? 1 : 0);
    }
  }
  return img;
}

describe('EnhanceMath', () => {
  it('AA strength 0 is identity', () => {
    const img = staircase();
    for (let y = 0; y < 24; y++) {
      for (let x = 0; x < 24; x++) {
        const out = edgeAa(img, x, y, 0);
        const src = img.sample(x + 0.5, y + 0.5);
        for (let c = 0; c < 3; c++) {
          expect(out[c]).toBeCloseTo(src[c], 5);
        }
      }
    }
  });

  it('AA leaves flat areas alone', () => {
    const img = new CpuImg(16, 16);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        img.px(x, y, 0.4, 0.5, 0.6);
      }
    }
    const out = edgeAa(img, 8, 8, 1);
    expect(out[0]).toBeCloseTo(0.4, 5);
    expect(out[2]).toBeCloseTo(0.6, 5);
  });

  it('AA leaves low contrast texture alone', () => {
    const img = new CpuImg(16, 16);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        img.gray(x, y, 0.5 + 0.01 * ((x + y) % 2));
      }
    }
    const out = edgeAa(img, 8, 8, 1);
    expect(out[0]).toBeCloseTo(img.sample(8.5, 8.5)[0], 5);
  });

  it('AA softens staircase corners and nothing far away', () => {
    const img = staircase();
    let softened = 0;
    for (let y = 4; y < 20; y++) {
      const x = y + 1;
      const v = edgeAa(img, x, y, 1)[0];
      if (v < 0.95 && v > 0) softened++;
    }
    expect(softened).toBeGreaterThanOrEqual(12);
    expect(edgeAa(img, 20, 3, 1)[0]).toBeCloseTo(1, 5);
    expect(edgeAa(img, 3, 20, 1)[0]).toBeCloseTo(0, 5);
  });

  it('AA output stays inside image range', () => {
    const img = staircase();
    for (let y = 0; y < 24; y++) {
      for (let x = 0; x < 24; x++) {
        const v = edgeAa(img, x, y, 1)[0];
        expect(v).toBeGreaterThanOrEqual(-1e-5);
        expect(v).toBeLessThanOrEqual(1 + 1e-5);
      }
    }
  });

  it('vibrance 0 is identity and grey stays grey', () => {
    const c: RgbTuple = [0.5, 0.4, 0.3];
    const same = vibrance(c, 0);
    for (let i = 0; i < 3; i++) expect(same[i]).toBeCloseTo(c[i], 5);
    const g = vibrance([0.5, 0.5, 0.5], 1);
    for (let i = 0; i < 3; i++) expect(g[i]).toBeCloseTo(0.5, 5);
  });

  it('vibrance boosts muted more than vivid', () => {
    const muted: RgbTuple = [0.55, 0.5, 0.45];
    const vivid: RgbTuple = [0.9, 0.3, 0.1];
    const gainMuted = sat(vibrance(muted, 0.5)) / sat(muted);
    const gainVivid = sat(vibrance(vivid, 0.5)) / sat(vivid);
    expect(gainMuted).toBeGreaterThan(gainVivid);
    expect(gainMuted).toBeGreaterThan(1.3);
  });

  it('clarity 0 is identity and flat stays flat', () => {
    const c: RgbTuple = [0.5, 0.4, 0.3];
    const same = clarity(c, 0.1, 0);
    for (let i = 0; i < 3; i++) expect(same[i]).toBeCloseTo(c[i], 5);
    const flat = clarity(c, luma(c), 1);
    for (let i = 0; i < 3; i++) expect(flat[i]).toBeCloseTo(c[i], 5);
  });

  it('clarity pushes away from surroundings', () => {
    const bright = clarity([0.6, 0.6, 0.6], 0.5, 1);
    const dark = clarity([0.4, 0.4, 0.4], 0.5, 1);
    expect(bright[0]).toBeGreaterThan(0.6);
    expect(dark[0]).toBeLessThan(0.4);
  });
});
