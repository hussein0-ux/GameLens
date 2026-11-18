export const MODE_LOOK = 0;   // heavy single-pass look filters, no upscale
export const MODE_GPU = 1;    // GPU 2x upscale (GSR or Catmull-Rom + HyperReal fused pass) + LUT
export const MODE_NEURAL = 2; // NPU 2x upscale + LUT (falls back to GPU in browser)
export const MODE_AUTO = 3;   // neural while it holds the frame rate, GPU otherwise
export const MODE_DLSS5 = 4;  // DLSS-5 equivalent: 3D-guided neural reconstruction + TAAU + frame gen

export const UPSCALER_CATMULL = 0; // Catmull-Rom + adaptive sharpen + HyperReal fused pass
export const UPSCALER_GSR = 1;     // Qualcomm Snapdragon Game Super Resolution 1 (edge-directed + HyperReal)

export type ModeId =
  | typeof MODE_LOOK
  | typeof MODE_GPU
  | typeof MODE_NEURAL
  | typeof MODE_AUTO
  | typeof MODE_DLSS5;
export type UpscalerId = typeof UPSCALER_CATMULL | typeof UPSCALER_GSR;

export interface PresetValues {
  sharpen: number;
  clarity: number;
  microRelief: number;
  dehaze: number;
  shadowLift: number;
  contrast: number;
  saturation: number;
  filmic: number;
  exposure: number;
  grade: number;
  vibrance: number;
  glow: number;
  vignette: number;
  chroma: number;
  motion: number;
  hdr: number;
  blackPt: number;
}

export interface LookState extends PresetValues {
  lut: number;
  mode: ModeId;
  upscaler: UpscalerId;
  upSharp: number;
  aa: number;
  upClarity: number;
  upVibrance: number;
  captureHalf: boolean;
  bypass: boolean;
  neuralLighting: number;
  frameGen: boolean;
}

export type PresetKey =
  | 'NATURAL'
  | 'VIVID'
  | 'CINEMATIC'
  | 'MONTAGE'
  | 'HDR_POP'
  | 'CRISP'
  | 'SD8_GEN3_HYPER';

export interface PresetDefinition {
  key: PresetKey;
  label: string;
  values: PresetValues;
}

export const PRESETS: Record<PresetKey, PresetDefinition> = {
  NATURAL: {
    key: 'NATURAL',
    label: 'Natural',
    values: {
      sharpen: 0.50,
      clarity: 0.25,
      microRelief: 0.25,
      dehaze: 0.10,
      shadowLift: 0.15,
      contrast: 1.03,
      saturation: 1.05,
      filmic: 0.20,
      exposure: 1.05,
      grade: 0.10,
      vibrance: 0.20,
      glow: 0.05,
      vignette: 0,
      chroma: 0,
      motion: 0,
      hdr: 0.15,
      blackPt: 0.10,
    },
  },
  VIVID: {
    key: 'VIVID',
    label: 'Vivid',
    values: {
      sharpen: 0.65,
      clarity: 0.40,
      microRelief: 0.45,
      dehaze: 0.22,
      shadowLift: 0.20,
      contrast: 1.08,
      saturation: 1.20,
      filmic: 0.30,
      exposure: 1.08,
      grade: 0.20,
      vibrance: 0.45,
      glow: 0.10,
      vignette: 0,
      chroma: 0,
      motion: 0,
      hdr: 0.35,
      blackPt: 0.20,
    },
  },
  CINEMATIC: {
    key: 'CINEMATIC',
    label: 'Cinematic',
    values: {
      sharpen: 0.60,
      clarity: 0.45,
      microRelief: 0.40,
      dehaze: 0.25,
      shadowLift: 0.18,
      contrast: 1.12,
      saturation: 1.10,
      filmic: 0.65,
      exposure: 1.10,
      grade: 0.60,
      vibrance: 0.35,
      glow: 0.30,
      vignette: 0.12,
      chroma: 0,
      motion: 0,
      hdr: 0.40,
      blackPt: 0.30,
    },
  },
  MONTAGE: {
    key: 'MONTAGE',
    label: 'Montage',
    values: {
      sharpen: 0.85,
      clarity: 0.55,
      microRelief: 0.60,
      dehaze: 0.30,
      shadowLift: 0.22,
      contrast: 1.15,
      saturation: 1.30,
      filmic: 0.50,
      exposure: 1.08,
      grade: 0.85,
      vibrance: 0.55,
      glow: 0.75,
      vignette: 0.35,
      chroma: 0.25,
      motion: 0.30,
      hdr: 0.25,
      blackPt: 0.15,
    },
  },
  HDR_POP: {
    key: 'HDR_POP',
    label: 'HDR Pop',
    values: {
      sharpen: 0.65,
      clarity: 0.45,
      microRelief: 0.55,
      dehaze: 0.32,
      shadowLift: 0.28,
      contrast: 1.10,
      saturation: 1.18,
      filmic: 0.55,
      exposure: 1.06,
      grade: 0.15,
      vibrance: 0.50,
      glow: 0.22,
      vignette: 0,
      chroma: 0,
      motion: 0,
      hdr: 0.85,
      blackPt: 0.35,
    },
  },
  CRISP: {
    key: 'CRISP',
    label: 'Crisp',
    values: {
      sharpen: 0.90,
      clarity: 0.60,
      microRelief: 0.55,
      dehaze: 0.25,
      shadowLift: 0.20,
      contrast: 1.08,
      saturation: 1.08,
      filmic: 0.25,
      exposure: 1.04,
      grade: 0.00,
      vibrance: 0.25,
      glow: 0.00,
      vignette: 0,
      chroma: 0,
      motion: 0,
      hdr: 0.30,
      blackPt: 0.15,
    },
  },
  SD8_GEN3_HYPER: {
    key: 'SD8_GEN3_HYPER',
    label: '8 Gen 3 Extreme',
    values: {
      sharpen: 0.94,
      clarity: 0.70,
      microRelief: 0.85,
      dehaze: 0.44,
      shadowLift: 0.38,
      contrast: 1.12,
      saturation: 1.15,
      filmic: 0.82,
      exposure: 1.06,
      grade: 0.24,
      vibrance: 0.45,
      glow: 0.38,
      vignette: 0.08,
      chroma: 0.05,
      motion: 0,
      hdr: 0.78,
      blackPt: 0.28,
    },
  },
};

export const DEFAULT_LOOK_STATE: LookState = {
  ...PRESETS.CRISP.values,
  lut: 1.0,
  mode: MODE_GPU,
  upscaler: UPSCALER_GSR,
  upSharp: 0.5,
  aa: 0.7,
  upClarity: 0.3,
  upVibrance: 0.25,
  captureHalf: true,
  bypass: false,
  neuralLighting: 0.8,
  frameGen: true,
};

export function applyPreset(current: LookState, presetKey: PresetKey): LookState {
  if (presetKey === 'SD8_GEN3_HYPER') {
    return {
      ...current,
      ...PRESETS.SD8_GEN3_HYPER.values,
      lut: 1.0,
      upSharp: 0.78,
      aa: 0.80,
      upClarity: 0.55,
      upVibrance: 0.35,
      captureHalf: true,
      bypass: false,
    };
  }
  return {
    ...current,
    ...PRESETS[presetKey].values,
  };
}

export interface SliderSpec {
  key: keyof LookState;
  label: string;
  min: number;
  max: number;
}

export const UPSCALE_SLIDERS: SliderSpec[] = [
  { key: 'upSharp', label: 'Upscale sharpness', min: 0, max: 1 },
  { key: 'aa', label: 'Edge anti-aliasing (GSR only)', min: 0, max: 1 },
  { key: 'upClarity', label: 'Enhance clarity (GSR only)', min: 0, max: 1 },
  { key: 'upVibrance', label: 'Enhance vibrance (GSR only)', min: 0, max: 1 },
];

export const LOOK_SLIDERS: SliderSpec[] = [
  { key: 'lut', label: 'LUT strength (HyperReal)', min: 0, max: 1 },
  { key: 'microRelief', label: 'Micro-texture relief (8G3)', min: 0, max: 1 },
  { key: 'dehaze', label: 'Atmospheric dehaze', min: 0, max: 1 },
  { key: 'shadowLift', label: 'Shadow detail recovery', min: 0, max: 1 },
  { key: 'hdr', label: 'ACES HDR look', min: 0, max: 1 },
  { key: 'filmic', label: 'ACES Filmic tone curve', min: 0, max: 1 },
  { key: 'glow', label: 'Specular bloom', min: 0, max: 1 },
  { key: 'sharpen', label: 'Sharpen', min: 0, max: 1 },
  { key: 'clarity', label: 'Clarity', min: 0, max: 1 },
  { key: 'exposure', label: 'Exposure', min: 0.8, max: 1.4 },
  { key: 'contrast', label: 'Contrast', min: 0.8, max: 1.4 },
  { key: 'saturation', label: 'Saturation', min: 0.5, max: 1.6 },
  { key: 'vibrance', label: 'Vibrance', min: 0, max: 1 },
  { key: 'grade', label: 'Teal/orange grade', min: 0, max: 1 },
  { key: 'blackPt', label: 'Deep blacks', min: 0, max: 1 },
  { key: 'vignette', label: 'Vignette', min: 0, max: 1 },
  { key: 'chroma', label: 'Chromatic fringe', min: 0, max: 1 },
  { key: 'motion', label: 'Motion blur', min: 0, max: 0.6 },
];

export const MODE_OPTIONS: { id: ModeId; label: string }[] = [
  { id: MODE_LOOK, label: 'Look filters only (heavy, no upscale)' },
  { id: MODE_GPU, label: 'GPU 2x SGSR + Fused HyperReal (120Hz SD 8 Gen 3)' },
  { id: MODE_NEURAL, label: 'Neural upscale + LUT (needs a model and a 1170x540 game)' },
  { id: MODE_AUTO, label: 'Auto: neural, falls back to GPU if it cannot hold the frame rate' },
  {
    id: MODE_DLSS5,
    label: 'DLSS-5 Equivalent: 3D-Guided Neural SR + Motion-Vector Frame Gen (120Hz)',
  },
];

export const UPSCALER_OPTIONS: { id: UpscalerId; label: string }[] = [
  { id: UPSCALER_GSR, label: 'Qualcomm GSR (edge-directed, made for games)' },
  { id: UPSCALER_CATMULL, label: 'Catmull-Rom + sharpen' },
];
