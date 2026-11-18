# GameLens

A real-time WebGL2 post-processing, 2x upscaling, and color grading studio for gameplay capture, rewritten in React + TypeScript + Vite from the original Android Kotlin + OpenGL ES 3.0/3.1 codebase.

This is **not** NVIDIA DLSS 5. GameLens is a screen-space post-processor: adaptive sharpening (CAS-style), clarity (micro-contrast), HDR local tone mapping, split-tone teal/orange grading, bloom glow, vignette, chromatic fringe, motion blur, 2x upscaling, and a 33³ HyperReal Rec.709 3D LUT.

## Presets & Look Controls

Presets: **Natural**, **Vivid**, **Cinematic**, **Montage**, **HDR Pop**, and **Crisp** (default).
Live sliders cover:

- **Upscaler & Enhance**: Upscale sharpness, Edge anti-aliasing (FXAA 3 Console, GSR only), Enhance clarity (GSR only), Enhance vibrance (GSR only)
- **Look & Grade**: LUT strength (HyperReal), Sharpen, Clarity, Exposure, Contrast, Saturation, Vibrance, Tone curve, Teal/orange grade, HDR look, Deep blacks, Glow, Vignette, Chromatic fringe, and Motion blur (ping-pong framebuffer blend)

## Pipeline Modes

- **Look filters only**: Full single-pass look shader (CAS sharpen, clarity, two-ring bloom, HDR tone mapping, S-curve, split-tone grade, vignette, chromatic fringe, motion blur, HyperReal 3D LUT).
- **GPU upscale + LUT**:
  - **Qualcomm GSR**: WebGL2 port of Snapdragon Game Super Resolution v1 (`sgsr1_mobile.frag`, BSD-3-Clause) preceded by FXAA 3 Console edge anti-aliasing.
  - **Catmull-Rom + sharpen**: 9-tap bicubic Catmull-Rom 2x upscaler with deringing clamp and CAS sharpening.
- **Neural upscale + LUT / Auto**: Inspects quantized planar NCHW `int8` `.tflite` / `.zip` models and tracks `NeuralGovernor` hysteresis (`< 110 fps` for 2s drops to GPU; `≥ 118 fps` for 5s recovers), falling back to the GPU upscaler in browser environments without a Qualcomm Hexagon NPU delegate.

## Sources & Comparison

- Built-in 16:9 gameplay scenes (**Tactical Arena** and **Alpine RPG**)
- Local gameplay **Video / Image upload**
- Live **Screen / App Capture** via `navigator.mediaDevices.getDisplayMedia`
- Interactive **Before / After Split Wipe** and **2x Pixel Inspect** loupe
- Optional **Capture at half screen size** downsampler to evaluate 2x reconstruction

## Development

```bash
npm install
npm run dev      # Starts Vite dev server on http://0.0.0.0:3000
npm run build    # Production bundle
npm run lint     # TypeScript type-check (tsc --noEmit)
npm test         # Vitest unit suite (CubeLut, EnhanceMath, NeuralGovernor)
```
