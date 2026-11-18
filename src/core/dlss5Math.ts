/**
 * Pure math utilities for the DLSS-5 equivalent pipeline:
 * - Halton (2, 3) 16-phase sub-pixel jitter sequence
 * - Minimal column-major 4x4 matrix math for G-buffer MVP + PrevMVP motion vectors
 * - Three-tier 120fps (8.33 ms) FrameBudgetGovernor
 */

export function halton(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

/** Returns sub-pixel jitter in [-0.5, 0.5] pixels for 16-phase TAAU accumulation. */
export function getHaltonJitter(frameIndex: number): [number, number] {
  const phase = (frameIndex % 16) + 1;
  return [halton(phase, 2) - 0.5, halton(phase, 3) - 0.5];
}

export type Mat4 = Float32Array;

export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4];
    const b1 = b[c * 4 + 1];
    const b2 = b[c * 4 + 2];
    const b3 = b[c * 4 + 3];
    out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

export function mat4Perspective(fovRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovRad / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

/** Applies sub-pixel camera jitter in clip space for half-resolution temporal super resolution. */
export function applySubpixelJitter(
  proj: Mat4,
  jitterPxX: number,
  jitterPxY: number,
  renderW: number,
  renderH: number,
): Mat4 {
  const out = new Float32Array(proj);
  out[8] += (2 * jitterPxX) / Math.max(1, renderW);
  out[9] += (2 * jitterPxY) / Math.max(1, renderH);
  return out;
}

export function mat4LookAt(
  eye: [number, number, number],
  center: [number, number, number],
  up: [number, number, number],
): Mat4 {
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len;
  zy /= len;
  zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len;
  xy /= len;
  xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const out = new Float32Array(16);
  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;
  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;
  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

export function mat4Transform(
  tx: number,
  ty: number,
  tz: number,
  rotY: number,
  sx = 1,
  sy = 1,
  sz = 1,
): Mat4 {
  const c = Math.cos(rotY);
  const s = Math.sin(rotY);
  const out = new Float32Array(16);
  out[0] = c * sx;
  out[2] = -s * sx;
  out[5] = sy;
  out[8] = s * sz;
  out[10] = c * sz;
  out[12] = tx;
  out[13] = ty;
  out[14] = tz;
  out[15] = 1;
  return out;
}

/**
 * Three-tier 120fps (8.33 ms budget) deadline governor:
 * - Tier 1: Full 3D-Guided Neural Reconstruction + Temporal SR + Frame Gen
 * - Tier 2: SGSR2-style Temporal SR (no neural synthesis pass)
 * - Tier 3: SGSR1 Spatial fallback
 */
export type DlssTier = 1 | 2 | 3;

export class FrameBudgetGovernor {
  tier: DlssTier = 1;
  private overBudgetFrames = 0;
  private underBudgetFrames = 0;

  constructor(
    private readonly highThresholdMs = 7.5,
    private readonly recoverThresholdMs = 5.2,
    private readonly downFrames = 8,
    private readonly upFrames = 24,
  ) {}

  onFrameTime(frameMs: number): DlssTier {
    if (frameMs > this.highThresholdMs) {
      this.overBudgetFrames++;
      this.underBudgetFrames = 0;
      if (this.overBudgetFrames >= this.downFrames && this.tier < 3) {
        this.tier = (this.tier + 1) as DlssTier;
        this.overBudgetFrames = 0;
      }
    } else if (frameMs < this.recoverThresholdMs) {
      this.underBudgetFrames++;
      this.overBudgetFrames = 0;
      if (this.underBudgetFrames >= this.upFrames && this.tier > 1) {
        this.tier = (this.tier - 1) as DlssTier;
        this.underBudgetFrames = 0;
      }
    } else {
      this.overBudgetFrames = 0;
      this.underBudgetFrames = 0;
    }
    return this.tier;
  }

  reset(): void {
    this.tier = 1;
    this.overBudgetFrames = 0;
    this.underBudgetFrames = 0;
  }
}
