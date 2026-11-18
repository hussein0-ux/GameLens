# GameLens (React + WebGL2)

A real-time post-processing and GPU upscaling filter for gameplay capture, rewritten from the original Android (Kotlin + OpenGL ES 3.0) project into a React + TypeScript + WebGL2 web application.

## Features

- **Live Capture & Clip Preview**: Share a single game window or browser tab via `navigator.mediaDevices.getDisplayMedia()`, load a local video/image clip, or tune live on the built-in sample gameplay frame.
- **Four Processing Modes**:
  - **Look filters only**: CAS-style adaptive sharpening, 4-tap clarity, 8-tap + wide-ring HDR glow, local tone-mapping HDR look, exposure, filmic S-curve, contrast, teal/orange split-tone grade, vibrance, saturation, black-point dehaze, vignette, chromatic fringe, frame-blend motion blur, and HyperReal 3D LUT.
  - **GPU upscale + LUT**: Chooses between **Qualcomm Snapdragon Game Super Resolution 1 (SGSR1)** with FXAA3 edge anti-aliasing, clarity, and vibrance, or **Catmull-Rom 2x** bicubic upscale with deringing and CAS sharpening, ending in the 33³ HyperReal Rec709 3D LUT (`public/hyperreal_rec709_33.cube`).
  - **Neural upscale + LUT / Auto**: Validates `.tflite` / `.zip` int8 planar NCHW models and tracks FPS hysteresis via `NeuralGovernor`, falling back to GPU upscaling in browser environments where Qualcomm Hexagon NPU hardware is unavailable.
- **Six Look Presets**: `Natural`, `Vivid`, `Cinematic`, `Montage`, `HDR Pop`, and `Crisp` (default).
- **Crash Log Diagnostics**: Records uncaught runtime or WebGL errors to `localStorage` with copy-and-clear controls.

## Development

```bash
npm install
npm run dev     # Starts Vite on 0.0.0.0:3000
npm run lint    # TypeScript typecheck
npm test        # Vitest unit tests (CubeLut, EnhanceMath, NeuralGovernor)
npm run build   # Production bundle
```

## Known Limitations

- Browser environments cannot invoke Android `MediaProjection`, `TYPE_ACCESSIBILITY_OVERLAY`, or Qualcomm Hexagon NPU FastRPC (`libcdsprpc.so` / `libQnnHtp.so`). Neural/Auto modes inspect and validate `.tflite` models and fall back to GPU upscaling.
- Third-party Qualcomm SGSR1 license is preserved in `public/SGSR_LICENSE.txt`.
