/**
 * CPU reference for the Enhance math in GlPipeline (GSR_COPY_FRAGMENT_SRC edge AA) and
 * sgsr1_mobile.frag (clarity, vibrance). The shaders are line-by-line translations of these functions.
 */
export class Img {
  readonly w: number;
  readonly h: number;
  readonly rgb: Float32Array;

  constructor(w: number, h: number, rgb?: Float32Array) {
    this.w = w;
    this.h = h;
    this.rgb = rgb ?? new Float32Array(w * h * 3);
  }

  px(x: number, y: number, r: number, g: number, b: number): void {
    const i = (y * this.w + x) * 3;
    this.rgb[i] = r;
    this.rgb[i + 1] = g;
    this.rgb[i + 2] = b;
  }

  gray(x: number, y: number, v: number): void {
    this.px(x, y, v, v, v);
  }

  private at(x: number, y: number, c: number): number {
    const cx = Math.min(Math.max(x, 0), this.w - 1);
    const cy = Math.min(Math.max(y, 0), this.h - 1);
    return this.rgb[(cy * this.w + cx) * 3 + c];
  }

  /** Bilinear sample at texel-centre coordinates, like the GPU's GL_LINEAR with clamp to edge. */
  sample(px: number, py: number): [number, number, number] {
    const fx = px - 0.5;
    const fy = py - 0.5;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = fx - x0;
    const ty = fy - y0;
    const out: [number, number, number] = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const top = this.at(x0, y0, c) * (1 - tx) + this.at(x0 + 1, y0, c) * tx;
      const bot = this.at(x0, y0 + 1, c) * (1 - tx) + this.at(x0 + 1, y0 + 1, c) * tx;
      out[c] = top * (1 - ty) + bot * ty;
    }
    return out;
  }
}

export function luma(c: readonly [number, number, number]): number {
  return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
}

function clamp01(v: number): number {
  return Math.min(Math.max(v, 0), 1);
}

/** FXAA 3 "console" directional blur, skipped on low-contrast pixels, mixed in by strength. */
export function edgeAa(img: Img, x: number, y: number, strength: number): [number, number, number] {
  const cx = x + 0.5;
  const cy = y + 0.5;
  const m = img.sample(cx, cy);
  if (strength <= 0) return m;

  const nw = luma(img.sample(cx - 1, cy - 1));
  const ne = luma(img.sample(cx + 1, cy - 1));
  const sw = luma(img.sample(cx - 1, cy + 1));
  const se = luma(img.sample(cx + 1, cy + 1));
  const lm = luma(m);
  const lumaMin = Math.min(lm, Math.min(Math.min(nw, ne), Math.min(sw, se)));
  const lumaMax = Math.max(lm, Math.max(Math.max(nw, ne), Math.max(sw, se)));
  if (lumaMax - lumaMin < Math.max(0.0312, lumaMax * 0.125)) return m;

  let dx = -((nw + ne) - (sw + se));
  let dy = (nw + sw) - (ne + se);
  const dirReduce = Math.max((nw + ne + sw + se) * 0.25 * 0.125, 1 / 128);
  const rcp = 1 / (Math.min(Math.abs(dx), Math.abs(dy)) + dirReduce);
  dx = Math.min(Math.max(dx * rcp, -8), 8);
  dy = Math.min(Math.max(dy * rcp, -8), 8);

  const a1 = img.sample(cx + dx * (1 / 3 - 0.5), cy + dy * (1 / 3 - 0.5));
  const a2 = img.sample(cx + dx * (2 / 3 - 0.5), cy + dy * (2 / 3 - 0.5));
  const b1 = img.sample(cx - dx * 0.5, cy - dy * 0.5);
  const b2 = img.sample(cx + dx * 0.5, cy + dy * 0.5);

  const rgbA: [number, number, number] = [
    0.5 * (a1[0] + a2[0]),
    0.5 * (a1[1] + a2[1]),
    0.5 * (a1[2] + a2[2]),
  ];
  const rgbB: [number, number, number] = [
    rgbA[0] * 0.5 + 0.25 * (b1[0] + b2[0]),
    rgbA[1] * 0.5 + 0.25 * (b1[1] + b2[1]),
    rgbA[2] * 0.5 + 0.25 * (b1[2] + b2[2]),
  ];
  const lb = luma(rgbB);
  const aa = lb < lumaMin || lb > lumaMax ? rgbA : rgbB;
  return [
    m[0] + (aa[0] - m[0]) * strength,
    m[1] + (aa[1] - m[1]) * strength,
    m[2] + (aa[2] - m[2]) * strength,
  ];
}

/** Pushes muted colours further than vivid ones; greys stay grey. */
export function vibrance(
  c: readonly [number, number, number],
  amount: number,
): [number, number, number] {
  const sat = Math.max(c[0], c[1], c[2]) - Math.min(c[0], c[1], c[2]);
  const k = 1 + amount * (1 - sat);
  const l = luma(c);
  return [
    clamp01(l + (c[0] - l) * k),
    clamp01(l + (c[1] - l) * k),
    clamp01(l + (c[2] - l) * k),
  ];
}

/** Local contrast: adds the difference between the pixel and a wide blur of its surroundings. */
export function clarity(
  c: readonly [number, number, number],
  blurLuma: number,
  amount: number,
): [number, number, number] {
  const d = (luma(c) - blurLuma) * amount;
  return [clamp01(c[0] + d), clamp01(c[1] + d), clamp01(c[2] + d)];
}
