export const MODE_LOOK = 0;
export const MODE_GPU = 1;
export const MODE_NEURAL = 2;
export const MODE_AUTO = 3;

export type PipelineMode =
  | typeof MODE_LOOK
  | typeof MODE_GPU
  | typeof MODE_NEURAL
  | typeof MODE_AUTO;

export const UPSCALER_CATMULL = 0;
export const UPSCALER_GSR = 1;

export type UpscalerType = typeof UPSCALER_CATMULL | typeof UPSCALER_GSR;

export interface LookPresetValues {
  sharpen: number;
  clarity: number;
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

export interface LookPreset extends LookPresetValues {
  id: string;
  label: string;
  summary: string;
}

export interface LookSettingsState extends LookPresetValues {
  lut: number;
  mode: PipelineMode;
  upscaler: UpscalerType;
  upSharp: number;
  aa: number;
  upClarity: number;
  upVibrance: number;
  captureHalf: boolean;
  bypass: boolean;
}

export const PRESETS: LookPreset[] = [
  {
    id: 'natural',
    label: 'Natural',
    summary: 'Balanced clarity and subtle contrast',
    sharpen: 0.5,
    clarity: 0.25,
    contrast: 1.03,
    saturation: 1.05,
    filmic: 0.2,
    exposure: 1.05,
    grade: 0.1,
    vibrance: 0.2,
    glow: 0.05,
    vignette: 0,
    chroma: 0,
    motion: 0,
    hdr: 0.15,
    blackPt: 0.1,
  },
  {
    id: 'vivid',
    label: 'Vivid',
    summary: 'Punchy color and clean detail lift',
    sharpen: 0.65,
    clarity: 0.4,
    contrast: 1.08,
    saturation: 1.2,
    filmic: 0.3,
    exposure: 1.08,
    grade: 0.2,
    vibrance: 0.45,
    glow: 0.1,
    vignette: 0,
    chroma: 0,
    motion: 0,
    hdr: 0.35,
    blackPt: 0.2,
  },
  {
    id: 'cinematic',
    label: 'Cinematic',
    summary: 'Teal shadows, warm highlights, soft bloom',
    sharpen: 0.6,
    clarity: 0.45,
    contrast: 1.12,
    saturation: 1.1,
    filmic: 0.45,
    exposure: 1.1,
    grade: 0.6,
    vibrance: 0.35,
    glow: 0.3,
    vignette: 0,
    chroma: 0,
    motion: 0,
    hdr: 0.3,
    blackPt: 0.3,
  },
  {
    id: 'montage',
    label: 'Montage',
    summary: 'Edit-style grade, bloom, fringe, motion blur',
    sharpen: 0.85,
    clarity: 0.55,
    contrast: 1.15,
    saturation: 1.3,
    filmic: 0.4,
    exposure: 1.08,
    grade: 0.85,
    vibrance: 0.55,
    glow: 0.75,
    vignette: 0.35,
    chroma: 0.25,
    motion: 0.3,
    hdr: 0,
    blackPt: 0,
  },
  {
    id: 'hdr_pop',
    label: 'HDR Pop',
    summary: 'Shadow lift, highlight punch, deep blacks',
    sharpen: 0.65,
    clarity: 0.45,
    contrast: 1.1,
    saturation: 1.18,
    filmic: 0.25,
    exposure: 1.06,
    grade: 0.15,
    vibrance: 0.5,
    glow: 0.15,
    vignette: 0,
    chroma: 0,
    motion: 0,
    hdr: 0.8,
    blackPt: 0.35,
  },
  {
    id: 'crisp',
    label: 'Crisp',
    summary: 'Edge sharpening first, zero bloom or blur',
    sharpen: 0.9,
    clarity: 0.6,
    contrast: 1.08,
    saturation: 1.08,
    filmic: 0.15,
    exposure: 1.04,
    grade: 0.0,
    vibrance: 0.25,
    glow: 0.0,
    vignette: 0,
    chroma: 0,
    motion: 0,
    hdr: 0.25,
    blackPt: 0.15,
  },
];

export const DEFAULT_LOOK_SETTINGS: LookSettingsState = {
  sharpen: 0.9,
  clarity: 0.6,
  contrast: 1.08,
  saturation: 1.08,
  filmic: 0.15,
  exposure: 1.04,
  grade: 0.0,
  vibrance: 0.25,
  glow: 0.0,
  vignette: 0.0,
  chroma: 0.0,
  motion: 0.0,
  hdr: 0.25,
  blackPt: 0.15,
  lut: 1.0,
  mode: MODE_GPU,
  upscaler: UPSCALER_GSR,
  upSharp: 0.5,
  aa: 0.7,
  upClarity: 0.3,
  upVibrance: 0.25,
  captureHalf: true,
  bypass: false,
};

export interface SliderSpec {
  key: keyof LookSettingsState;
  label: string;
  min: number;
  max: number;
  step: number;
  group: 'upscaler' | 'look';
}

export const UPSCALER_SLIDERS: SliderSpec[] = [
  { key: 'upSharp', label: 'Upscale sharpness', min: 0, max: 1, step: 0.01, group: 'upscaler' },
  { key: 'aa', label: 'Edge anti-aliasing (GSR only)', min: 0, max: 1, step: 0.01, group: 'upscaler' },
  { key: 'upClarity', label: 'Enhance clarity (GSR only)', min: 0, max: 1, step: 0.01, group: 'upscaler' },
  { key: 'upVibrance', label: 'Enhance vibrance (GSR only)', min: 0, max: 1, step: 0.01, group: 'upscaler' },
];

export const LOOK_SLIDERS: SliderSpec[] = [
  { key: 'lut', label: 'LUT strength (HyperReal)', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'sharpen', label: 'Sharpen', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'clarity', label: 'Clarity', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'exposure', label: 'Exposure', min: 0.8, max: 1.4, step: 0.01, group: 'look' },
  { key: 'contrast', label: 'Contrast', min: 0.8, max: 1.4, step: 0.01, group: 'look' },
  { key: 'saturation', label: 'Saturation', min: 0.5, max: 1.6, step: 0.01, group: 'look' },
  { key: 'vibrance', label: 'Vibrance', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'filmic', label: 'Tone curve', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'grade', label: 'Teal/orange grade', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'hdr', label: 'HDR look', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'blackPt', label: 'Deep blacks', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'glow', label: 'Glow', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'vignette', label: 'Vignette', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'chroma', label: 'Chromatic fringe', min: 0, max: 1, step: 0.01, group: 'look' },
  { key: 'motion', label: 'Motion blur', min: 0, max: 0.6, step: 0.01, group: 'look' },
];

export const MODE_OPTIONS: { value: PipelineMode; label: string; detail: string }[] = [
  {
    value: MODE_LOOK,
    label: 'Look filters only (heavy, no upscale)',
    detail: 'Single-pass CAS sharpen, clarity, HDR tone map, split-tone grade, bloom, vignette, chromatic fringe, motion blur & LUT',
  },
  {
    value: MODE_GPU,
    label: 'GPU upscale + LUT (fast)',
    detail: 'Qualcomm SGSR 1 edge-directed or Catmull-Rom 2x upscaling + HyperReal 33³ 3D LUT',
  },
  {
    value: MODE_NEURAL,
    label: 'Neural upscale + LUT (needs a model and a 1170x540 game)',
    detail: 'Inspects planar NCHW int8 .tflite models; falls back to GPU upscaler in browser environments without Hexagon NPU',
  },
  {
    value: MODE_AUTO,
    label: 'Auto: neural, falls back to GPU if it cannot hold the frame rate',
    detail: 'Uses NeuralGovernor hysteresis (drops after 2s < 110 fps, recovers after 5s ≥ 118 fps)',
  },
];

export const UPSCALER_OPTIONS: { value: UpscalerType; label: string }[] = [
  { value: UPSCALER_GSR, label: 'Qualcomm GSR (edge-directed, made for games)' },
  { value: UPSCALER_CATMULL, label: 'Catmull-Rom + sharpen' },
];

export interface AiRuntimeState {
  modelName: string | null;
  status: string;
  stats: string;
  diag: string;
}
