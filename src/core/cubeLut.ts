/**
 * A 3D colour lookup table from an Adobe/Resolve ".cube" file.
 * `rgb` holds size^3 entries of 3 floats each, red index varying fastest (the order the file uses),
 * normalised to 0..1 by the file's DOMAIN_MIN / DOMAIN_MAX.
 */
export interface CubeLut {
  size: number;
  rgb: Float32Array;
}

/** Parses the text of a .cube file. Throws Error with a readable reason on bad input. */
export function parseCube(text: string): CubeLut {
  let size = 0;
  let domainMin: [number, number, number] = [0, 0, 0];
  let domainMax: [number, number, number] = [1, 1, 1];
  const values: number[] = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.length === 0 || line.startsWith('#') || line.startsWith('TITLE')) continue;
    const parts = line.split(/\s+/);
    switch (parts[0]) {
      case 'LUT_3D_SIZE':
        size = Number.parseInt(parts[1], 10);
        break;
      case 'LUT_1D_SIZE':
        throw new Error('1D LUTs are not supported');
      case 'DOMAIN_MIN':
        domainMin = [
          Number.parseFloat(parts[1]),
          Number.parseFloat(parts[2]),
          Number.parseFloat(parts[3]),
        ];
        break;
      case 'DOMAIN_MAX':
        domainMax = [
          Number.parseFloat(parts[1]),
          Number.parseFloat(parts[2]),
          Number.parseFloat(parts[3]),
        ];
        break;
      default: {
        if (parts.length !== 3) {
          throw new Error(`Unexpected line in .cube file: ${line}`);
        }
        for (let c = 0; c < 3; c++) {
          const v = Number.parseFloat(parts[c]);
          if (Number.isNaN(v)) {
            throw new Error(`Invalid float in .cube file: ${line}`);
          }
          values.push((v - domainMin[c]) / (domainMax[c] - domainMin[c]));
        }
        break;
      }
    }
  }

  if (!Number.isInteger(size) || size < 2 || size > 256) {
    throw new Error(`Missing or invalid LUT_3D_SIZE (got ${size})`);
  }
  const expected = size * size * size;
  if (values.length !== expected * 3) {
    throw new Error(`expected ${expected} entries, got ${values.length / 3}`);
  }
  return { size, rgb: new Float32Array(values) };
}
