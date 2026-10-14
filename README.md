# GameLens

A real-time "better graphics" filter for Android gameplay, written in Kotlin + OpenGL ES 3.0.
Built with a Samsung Galaxy S24 Ultra (Android 14+) in mind.

This is **not** NVIDIA DLSS 5. DLSS 5 needs engine data and RTX 50 Tensor Cores. GameLens is a
screen-space post-process: adaptive sharpening, clarity (micro-contrast), a filmic tone curve,
contrast and saturation. It can make footage look crisper and punchier, but it cannot invent
new lighting or materials.

## Looks (v0.3)

Presets in the app: **Natural**, **Vivid** (default) and **Cinematic**. Sliders cover sharpen, clarity,
exposure, contrast, saturation, vibrance, tone curve, teal/orange grade and glow. Exposure and the
gentler tone curve fix the dim look of v0.2.

## Montage preset (v0.5)

A fourth preset, **Montage**, aims at the look of edited gameplay clips: heavy teal/orange grade, strong two-ring
bloom, crisp sharpening, vignette, chromatic fringe and frame-blend motion blur. New sliders: **Vignette**,
**Chromatic fringe**, **Motion blur** (blends the previous output frame in; 0 disables the extra framebuffer pass).
Speed ramps, zoom punches and cuts are editing effects and are not possible in a live overlay.

## How it works

```
game app --MediaProjection--> VirtualDisplay --> SurfaceTexture (OES texture)
      --> fragment shader (sharpen / clarity / filmic / contrast / saturation)
      --> full-screen click-through overlay (SurfaceView)
```

Everything stays on the GPU, so there are no CPU frame copies.

## Build and run

1. Install Android Studio and open the `GameLens` folder. Let Gradle sync (it will create the wrapper).
2. Enable Developer options and USB debugging on the S24 Ultra, plug it in, press Run.
3. In the app: press **Start**. The first time, Android opens Accessibility settings: turn on
   **GameLens overlay** (under Installed apps). If it is greyed out, open App info for GameLens,
   tap the ⋮ menu and choose **Allow restricted settings**, then try again. Press **Start** again.
   The accessibility service does nothing except let the overlay be a trusted window, so taps pass
   through to the game instead of being blocked.
4. When Android asks what to share, choose **A single app** and select your game.
5. Switch to the game. To change the look, switch back to GameLens and move the sliders; changes
   apply live. Use the notification's **Stop** button to end.

## Why "single app" matters

In whole-screen capture the overlay would be captured back into itself (a feedback loop that
over-sharpens every frame). Single-app capture only includes the chosen app's window, so the
overlay stays out of the capture.

## AI upscale prototype (v0.4, experimental)

Runs QuickSRNetSmall (ONNX, 128x128 -> 512x512) on the Snapdragon NPU through ONNX Runtime + QNN.
Pipeline: capture -> GPU box-filter down to (cols*128 x rows*128) -> read back -> tiles through the NPU
-> float atlas texture -> draw to overlay. The look filters are bypassed in this mode.

1. Get the QuickSRNetSmall **ONNX** zip from Qualcomm AI Hub / Hugging Face (float I/O, as used by QuickSR Bench).
2. In the app: pick the model, tick **Enable AI upscale**, choose an input size, press Start.
3. Open your game. Return to GameLens to read the per-frame timing breakdown.
4. Use "Split" view to compare AI (left) against plain bilinear of the same small image (right).

Known issues of the prototype: tiles are processed independently, so faint seams at 128 px borders are
expected; stats are only visible in the app, not over the game; landscape only; the full-resolution
capture is deliberately shrunk, so this shows pipeline cost and AI-vs-bilinear quality, not a
real "game rendered at low resolution" gain.

## Known limitations

- Games that set `FLAG_SECURE` or block capture will show black.
- Some online games treat overlays or capture as cheating. Use at your own risk.
- Adds roughly one to two frames of latency and some GPU/battery cost. Fine for most games,
  not great for competitive shooters or rhythm games.
- Capture runs at native screen resolution. It is not an upscaler: it does not add real pixels.
- Not compiled or tested on a device in this session. Expect small fixes on first run.

## Where to go next

- Replace the fragment shader with a small learned model (TFLite GPU delegate or NCNN with Vulkan)
  for AI upscaling or denoising, ideally at a reduced render scale to protect frame rate.
- Add a floating quick-settings tile or bubble to change the look without leaving the game.
- Add per-game presets saved by package name.
