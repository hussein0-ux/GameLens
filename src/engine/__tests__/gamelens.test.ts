import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCube } from '../cubeLut';
import { clarity, edgeAa, Img, luma, vibrance } from '../enhanceMath';
import { NeuralGovernor } from '../neuralGovernor';

describe('CubeLut (ported from CubeLutTest.kt)', () => {
  const identity2 = [
    'TITLE "identity"',
    'LUT_3D_SIZE 2',
    '0 0 0',
    '1 0 0',
    '0 1 0',
    '1 1 0',
    '0 0 1',
    '1 0 1',
    '0 1 1',
    '1 1 1',
  ].join('\n');

  it('parses size and keeps red as the fastest axis', () => {
    const lut = parseCube(identity2);
    expect(lut.size).toBe(2);
    expect(lut.rgb.length).toBe(2 * 2 * 2 * 3);
    expect(Array.from(lut.rgb.slice(3, 6))).toEqual([1, 0, 0]);
    expect(Array.from(lut.rgb.slice(6, 9))).toEqual([0, 1, 0]);
    expect(Array.from(lut.rgb.slice(21, 24))).toEqual([1, 1, 1]);
  });

  it('ignores comments, blank lines, and title', () => {
    const text = '# made by hand\n\nTITLE "x"\n' + identity2.split('\n').slice(1).join('\n');
    expect(parseCube(text).size).toBe(2);
  });

  it('rejects wrong entry count', () => {
    const short = identity2.split('\n').slice(0, -1).join('\n');
    expect(() => parseCube(short)).toThrow(/expected 8.*got 7/);
  });

  it('rejects missing size and 1D tables', () => {
    expect(() => parseCube('0 0 0\n1 1 1')).toThrow();
    expect(() => parseCube('LUT_1D_SIZE 2\n0 0 0\n1 1 1')).toThrow(/1D LUTs/);
  });

  it('normalises custom domain', () => {
    const lines = identity2
      .split('\n')
      .slice(2)
      .map((l) =>
        l
          .split(' ')
          .map((n) => (Number(n) * 2).toString())
          .join(' '),
      )
      .join('\n');
    const text = `LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n${lines}`;
    const lut = parseCube(text);
    expect(Array.from(lut.rgb.slice(21, 24))).toEqual([1, 1, 1]);
  });

  it('parses the shipped hyperreal_rec709_33.cube', () => {
    const cubePath = path.resolve(process.cwd(), 'public/assets/hyperreal_rec709_33.cube');
    const lut = parseCube(fs.readFileSync(cubePath, 'utf8'));
    expect(lut.size).toBe(33);
    expect(lut.rgb.length).toBe(33 * 33 * 33 * 3);
    expect(lut.rgb[1]).toBeCloseTo(0.000421, 5);
    expect(lut.rgb[lut.rgb.length - 1]).toBeCloseTo(0.998789, 5);
  });
});

describe('EnhanceMath (ported from EnhanceMathTest.kt)', () => {
  function staircase(): Img {
    const img = new Img(24, 24);
    for (let y = 0; y < 24; y++) {
      for (let x = 0; x < 24; x++) {
        img.gray(x, y, x > y ? 1 : 0);
      }
    }
    return img;
  }

  it('aaStrengthZeroIsIdentity', () => {
    const img = staircase();
    for (let y = 0; y < 24; y++) {
      for (let x = 0; x < 24; x++) {
        const out = edgeAa(img, x, y, 0);
        const src = img.sample(x + 0.5, y + 0.5);
        for (let c = 0; c < 3; c++) {
          expect(out[c]).toBeCloseTo(src[c], 6);
        }
      }
    }
  });

  it('aaLeavesFlatAreasAlone', () => {
    const img = new Img(16, 16);
    for (let y = 0; y < 16; y++) {
      for (let x = 0; x < 16; x++) {
        img.px(x, y, 0.4, 0.5, 0.6);
      }
    }
    const out = edgeAa(img, 8, 8, 1);
    expect(out[0]).toBeCloseTo(0.4, 5);
    expect(out[2]).toBeCloseTo(0.6, 5);
  });

  it('aaSoftensStaircaseCornersAndNothingFarAway', () => {
    const img = staircase();
    let softened = 0;
    for (let y = 4; y < 20; y++) {
      const v = edgeAa(img, y + 1, y, 1)[0];
      if (v < 0.95 && v > 0.0) softened++;
    }
    expect(softened).toBeGreaterThanOrEqual(12);
    expect(edgeAa(img, 20, 3, 1)[0]).toBeCloseTo(1, 5);
    expect(edgeAa(img, 3, 20, 1)[0]).toBeCloseTo(0, 5);
  });

  it('vibranceBoostsMutedMoreThanVivid', () => {
    const sat = (c: readonly [number, number, number]) =>
      Math.max(c[0], c[1], c[2]) - Math.min(c[0], c[1], c[2]);
    const muted = [0.55, 0.5, 0.45] as const;
    const vivid = [0.9, 0.3, 0.1] as const;
    const gainMuted = sat(vibrance(muted, 0.5)) / sat(muted);
    const gainVivid = sat(vibrance(vivid, 0.5)) / sat(vivid);
    expect(gainMuted).toBeGreaterThan(gainVivid);
    expect(gainMuted).toBeGreaterThan(1.3);
  });

  it('clarityPushesAwayFromSurroundings', () => {
    const c = [0.5, 0.4, 0.3] as const;
    expect(clarity(c, luma(c), 1)[0]).toBeCloseTo(0.5, 5);
    expect(clarity([0.6, 0.6, 0.6], 0.5, 1)[0]).toBeGreaterThan(0.6);
    expect(clarity([0.4, 0.4, 0.4], 0.5, 1)[0]).toBeLessThan(0.4);
  });
});

describe('NeuralGovernor (ported from NeuralGovernorTest.kt)', () => {
  const feed = (g: NeuralGovernor, ...fps: number[]) => fps.forEach((f) => g.onSecond(f));

  it('starts on neural and stays while fast', () => {
    const g = new NeuralGovernor();
    feed(g, 119, 120, 118, 119);
    expect(g.useNeural).toBe(true);
  });

  it('one slow second does not switch', () => {
    const g = new NeuralGovernor();
    feed(g, 100, 119, 100, 119);
    expect(g.useNeural).toBe(true);
  });

  it('two slow seconds in a row switch to GPU', () => {
    const g = new NeuralGovernor();
    feed(g, 100, 105);
    expect(g.useNeural).toBe(false);
  });

  it('returns to neural only after five fast seconds', () => {
    const g = new NeuralGovernor();
    feed(g, 90, 90);
    feed(g, 120, 120, 120, 120);
    expect(g.useNeural).toBe(false);
    feed(g, 120);
    expect(g.useNeural).toBe(true);
  });

  it('a slow second resets recovery', () => {
    const g = new NeuralGovernor();
    feed(g, 90, 90);
    feed(g, 120, 120, 120, 120, 110, 120, 120, 120, 120);
    expect(g.useNeural).toBe(false);
    feed(g, 120);
    expect(g.useNeural).toBe(true);
  });
});
