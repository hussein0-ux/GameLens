import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseCube } from '../cubeLut';

const identity2 = `
TITLE "identity"
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
`.trim();

describe('CubeLut', () => {
  it('parses size and keeps red as the fastest axis', () => {
    const lut = parseCube(identity2);
    expect(lut.size).toBe(2);
    expect(lut.rgb.length).toBe(2 * 2 * 2 * 3);
    expect(Array.from(lut.rgb.slice(3, 6))).toEqual([1, 0, 0]);
    expect(Array.from(lut.rgb.slice(6, 9))).toEqual([0, 1, 0]);
    expect(Array.from(lut.rgb.slice(21, 24))).toEqual([1, 1, 1]);
  });

  it('ignores comments, blank lines, and TITLE', () => {
    const text = `# made by hand\n\nTITLE "x"\n` + identity2.split('\n').slice(1).join('\n');
    expect(parseCube(text).size).toBe(2);
  });

  it('rejects wrong entry count', () => {
    const short = identity2.split('\n').slice(0, -1).join('\n');
    expect(() => parseCube(short)).toThrow(/expected 8.*got 7/);
  });

  it('rejects missing size and 1D tables', () => {
    expect(() => parseCube('0 0 0\n1 1 1')).toThrow();
    expect(() => parseCube('LUT_1D_SIZE 2\n0 0 0\n1 1 1')).toThrow();
  });

  it('normalises custom domain', () => {
    const lines = identity2.split('\n').slice(2).map((l) =>
      l
        .split(' ')
        .map((n) => String(Number.parseFloat(n) * 2))
        .join(' '),
    );
    const text = `LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n${lines.join('\n')}`;
    const lut = parseCube(text);
    expect(Array.from(lut.rgb.slice(21, 24))).toEqual([1, 1, 1]);
  });

  it('parses the shipped HyperReal .cube LUT', () => {
    const raw = readFileSync(resolve(process.cwd(), 'public/hyperreal_rec709_33.cube'), 'utf8');
    const lut = parseCube(raw);
    expect(lut.size).toBe(33);
    expect(lut.rgb.length).toBe(33 * 33 * 33 * 3);
    expect(lut.rgb[1]).toBeCloseTo(0.000421, 5);
    expect(lut.rgb[lut.rgb.length - 1]).toBeCloseTo(0.998789, 5);
  });
});
