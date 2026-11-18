/**
 * CPU reference for the Enhance math in GlPipeline (GSR_COPY_FRAGMENT_SRC edge AA) and
 * sgsr1_mobile.frag (clarity, vibrance). The WebGL2 shaders are line-by-line translations of these functions.
 */
export type RgbTuple = [number, number, number];

export class CpuImg {
  readonly rgb: Float32Array;

  constructor(readonly w: number, readonly h: number, rgb?: Float32Array) {
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
    const cx = Math.max(0, Math.min(this.w - 1, x));
    const cy = Math.max(0, Math.min(this.h - 1, y));
    return this.rgb[(cy * this.w + cx) * 3 + c];
  }

  /** Bilinear sample at texel-centre coordinates, like GL_LINEAR with clamp to edge. */
  sample(px: number, py: number): RgbTuple {
    const fx = px - 0.5;
    const fy = py - 0.5;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = fx - x0;
    const ty = fy - y0;
    const out: RgbTuple = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const top = this.at(x0, y0, c) * (1 - tx) + this.at(x0 + 1, y0, c) * tx;
      const bot = this.at(x0, y0 + 1, c) * (1 - tx) + this.at(x0 + 1, y0 + 1, c) * tx;
      out[c] = top * (1 - ty) + bot * ty;
    }
    return out;
  }
}

export function luma(c: RgbTuple): number {
  return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** FXAA 3 "console" directional blur, skipped on low-contrast pixels, mixed in by `strength`. */
export function edgeAa(img: CpuImg, x: number, y: number, strength: number): RgbTuple {
  const cx = x + 0.5;
  const cy = y + 0.5;
  const m = img.sample(cx, cy);
  if (strength <= 0) return m;
  const nw = luma(img.sample(cx - 1, cy - 1));
  const ne = luma(img.sample(cx + 1, cy - 1));
  const sw = luma(img.sample(cx - 1, cy + 1));
  const se = luma(img.sample(cx + 1, cy + 1));
  const lm = luma(m);
  const lumaMin = Math.min(lm, nw, ne, sw, se);
  const lumaMax = Math.max(lm, nw, ne, sw, se);
  if (lumaMax - lumaMin < Math.max(0.0312, lumaMax * 0.125)) return m;

  let dx = -((nw + ne) - (sw + se));
  let dy = (nw + sw) - (ne + se);
  const dirReduce = Math.max((nw + ne + sw + se) * 0.25 * 0.125, 1 / 128);
  const rcp = 1 / (Math.min(Math.abs(dx), Math.abs(dy)) + dirReduce);
  dx = clamp(dx * rcp, -8, 8);
  dy = clamp(dy * rcp, -8, 8);

  const a1 = img.sample(cx + dx * (1 / 3 - 0.5), cy + dy * (1 / 3 - 0.5));
  const a2 = img.sample(cx + dx * (2 / 3 - 0.5), cy + dy * (2 / 3 - 0.5));
  const b1 = img.sample(cx - dx * 0.5, cy - dy * 0.5);
  const b2 = img.sample(cx + dx * 0.5, cy + dy * 0.5);
  const rgbA: RgbTuple = [
    0.5 * (a1[0] + a2[0]),
    0.5 * (a1[1] + a2[1]),
    0.5 * (a1[2] + a2[2]),
  ];
  const rgbB: RgbTuple = [
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
export function vibrance(c: RgbTuple, amount: number): RgbTuple {
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
export function clarity(c: RgbTuple, blurLuma: number, amount: number): RgbTuple {
  const d = (luma(c) - blurLuma) * amount;
  return [clamp01(c[0] + d), clamp01(c[1] + d), clamp01(c[2] + d)];
}
