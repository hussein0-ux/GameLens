import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LOOK_STATE,
  LookState,
  MODE_DLSS5,
  MODE_GPU,
  MODE_LOOK,
  PRESETS,
  UPSCALER_GSR,
} from '../lookSettings';
import { estimateSd8Gen3Budget, SD8_GEN3_DEADLINE_MS } from '../sd8Gen3Profile';

describe('estimateSd8Gen3Budget', () => {
  it('keeps SD 8 Gen 3 HyperReal SGSR half-res comfortably under the 8.33 ms 120fps budget', () => {
    const state: LookState = {
      ...DEFAULT_LOOK_STATE,
      ...PRESETS.SD8_GEN3_HYPER.values,
      mode: MODE_GPU,
      upscaler: UPSCALER_GSR,
      captureHalf: true,
    };
    const est = estimateSd8Gen3Budget(state, 2400, 1080);
    expect(est.adreno750Ms).toBeLessThan(5.8);
    expect(est.headroomMs).toBeGreaterThan(2.5);
    expect(est.verdict).toBe('LOCKED_120');
    expect(est.projectedFps).toBe(120);
  });

  it('shows higher cost in heavy full-res Look mode than Half-Res GPU SGSR', () => {
    const gpuHalf = estimateSd8Gen3Budget(
      { ...DEFAULT_LOOK_STATE, mode: MODE_GPU, captureHalf: true },
      2560,
      1440,
    );
    const lookFull = estimateSd8Gen3Budget(
      { ...DEFAULT_LOOK_STATE, mode: MODE_LOOK, captureHalf: false, glow: 0.8, hdr: 0.8 },
      2560,
      1440,
    );
    expect(lookFull.adreno750Ms).toBeGreaterThan(gpuHalf.adreno750Ms);
    expect(lookFull.deadlineMs).toBe(SD8_GEN3_DEADLINE_MS);
  });

  it('reduces amortized cost in DLSS-5 mode when Frame Generation is enabled', () => {
    const noFg = estimateSd8Gen3Budget(
      { ...DEFAULT_LOOK_STATE, mode: MODE_DLSS5, frameGen: false, captureHalf: true },
      2400,
      1080,
    );
    const withFg = estimateSd8Gen3Budget(
      { ...DEFAULT_LOOK_STATE, mode: MODE_DLSS5, frameGen: true, captureHalf: true },
      2400,
      1080,
    );
    expect(withFg.adreno750Ms).toBeLessThan(noFg.adreno750Ms);
  });
});
