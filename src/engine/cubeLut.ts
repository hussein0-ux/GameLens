/**
 * A 3D colour lookup table from an Adobe/Resolve ".cube" file.
 * rgb holds size^3 entries of 3 floats each, red index varying fastest (the order the file uses),
 * normalised to 0..1 by the file's DOMAIN_MIN / DOMAIN_MAX.
 */
export interface CubeLut {
  size: number;
  rgb: Float32Array;
  title?: string;
}

/**
 * Parses the text of a .cube file.
 * Throws Error with a readable reason on invalid input.
 */
export function parseCube(text: string): CubeLut {
  let size = 0;
  let title: string | undefined;
  let domainMin: [number, number, number] = [0, 0, 0];
  let domainMax: [number, number, number] = [1, 1, 1];
  const values: number[] = [];

  const lines = text.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    if (line.startsWith('TITLE')) {
      const m = line.match(/^TITLE\s+"?(.*?)"?$/);
      if (m) title = m[1];
      continue;
    }

    const parts = line.split(/\s+/);
    const key = parts[0];

    if (key === 'LUT_3D_SIZE') {
      size = Number.parseInt(parts[1], 10);
      if (!Number.isFinite(size)) {
        throw new Error(`Invalid LUT_3D_SIZE: ${parts[1]}`);
      }
    } else if (key === 'LUT_1D_SIZE') {
      throw new Error('1D LUTs are not supported');
    } else if (key === 'DOMAIN_MIN') {
      domainMin = [
        Number.parseFloat(parts[1]),
        Number.parseFloat(parts[2]),
        Number.parseFloat(parts[3]),
      ];
    } else if (key === 'DOMAIN_MAX') {
      domainMax = [
        Number.parseFloat(parts[1]),
        Number.parseFloat(parts[2]),
        Number.parseFloat(parts[3]),
      ];
    } else {
      if (parts.length !== 3) {
        throw new Error(`Unexpected line in .cube file: ${line}`);
      }
      for (let c = 0; c < 3; c++) {
        const num = Number.parseFloat(parts[c]);
        if (!Number.isFinite(num)) {
          throw new Error(`Invalid number in .cube file: ${line}`);
        }
        const span = domainMax[c] - domainMin[c];
        values.push((num - domainMin[c]) / span);
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

  return {
    size,
    rgb: new Float32Array(values),
    title,
  };
}
