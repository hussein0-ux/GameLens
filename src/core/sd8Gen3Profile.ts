import {
  LookState,
  MODE_DLSS5,
  MODE_LOOK,
  UPSCALER_GSR,
} from './lookSettings';

export interface BudgetEstimate {
  adreno750Ms: number;
  deadlineMs: number;
  headroomMs: number;
  projectedFps: number;
  verdict: 'LOCKED_120' | 'TIGHT_120' | 'OVER_BUDGET';
  verdictLabel: string;
  gameRecommendation: string;
}

export const SD8_GEN3_DEADLINE_MS = 8.33;

/**
 * Estimates per-frame post-process + upscaler GPU cost on Qualcomm Snapdragon 8 Gen 3 (Adreno 750)
 * in milliseconds for a given output resolution and LookState configuration.
 */
export function estimateSd8Gen3Budget(
  settings: LookState,
  outWidth = 2400,
  outHeight = 1080,
): BudgetEstimate {
  const mp = Math.max(0.5, (outWidth * outHeight) / 1_000_000);
  const halfScaleFactor = settings.captureHalf ? 0.35 : 1.0;

  let ms = 0.25; // base swap / surface blit overhead

  if (settings.bypass) {
    ms += 0.2 * mp;
  } else if (settings.mode === MODE_LOOK) {
    // Heavy full-res multi-tap look shader
    ms += 1.95 * mp * halfScaleFactor;
    if (settings.glow > 0.05) ms += 0.45 * mp;
    if (settings.hdr > 0.05) ms += 0.35 * mp;
  } else if (settings.mode === MODE_DLSS5) {
    // Half-res MRT G-Buffer + TAAU + optional Frame Gen
    ms += 1.45 * mp * (settings.captureHalf ? 0.55 : 1.0);
    if (settings.neuralLighting > 0.05) ms += 0.65 * mp;
    if (settings.frameGen) {
      // Midpoint FG skips G-buffer every other frame -> amortized reduction
      ms = ms * 0.62 + 0.55 * mp;
    }
  } else {
    // MODE_GPU / MODE_NEURAL / MODE_AUTO -> fused SGSR1 or Catmull-Rom + HyperReal pass
    const copyPass = (settings.aa ?? 0) > 0.01 ? 0.32 * mp * halfScaleFactor : 0.12 * mp * halfScaleFactor;
    const upscaleBase = settings.upscaler === UPSCALER_GSR ? 0.62 * mp : 0.44 * mp;
    const hyperFused =
      ((settings.microRelief ?? 0) > 0.01 ? 0.14 * mp : 0) +
      ((settings.hdr ?? 0) > 0.01 || (settings.filmic ?? 0) > 0.01 ? 0.12 * mp : 0) +
      ((settings.glow ?? 0) > 0.01 ? 0.18 * mp : 0) +
      ((settings.dehaze ?? 0) > 0.01 || (settings.shadowLift ?? 0) > 0.01 ? 0.09 * mp : 0);
    ms += copyPass + upscaleBase + hyperFused;
  }

  const adreno750Ms = Number(ms.toFixed(2));
  const headroomMs = Number((SD8_GEN3_DEADLINE_MS - adreno750Ms).toFixed(2));
  const projectedFps = Math.min(120, Math.max(30, Math.round(1000 / Math.max(adreno750Ms + 3.8, 8.33))));

  let verdict: BudgetEstimate['verdict'] = 'LOCKED_120';
  let verdictLabel = 'Locked 120Hz headroom';
  if (adreno750Ms > SD8_GEN3_DEADLINE_MS) {
    verdict = 'OVER_BUDGET';
    verdictLabel = 'Exceeds 8.33 ms post budget';
  } else if (adreno750Ms > 5.8) {
    verdict = 'TIGHT_120';
    verdictLabel = 'Tight 120Hz margin (enable Half-Res)';
  }

  const gameRecommendation = settings.captureHalf
    ? 'CoD Mobile: Medium Graphics + Ultra FPS (120Hz) -> GameLens 2x SGSR HyperReal'
    : 'CoD Mobile: High Graphics + Ultra FPS (enable Half-Res for 2x SGSR speed)';

  return {
    adreno750Ms,
    deadlineMs: SD8_GEN3_DEADLINE_MS,
    headroomMs,
    projectedFps,
    verdict,
    verdictLabel,
    gameRecommendation,
  };
}
