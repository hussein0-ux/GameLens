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

## Modes (v0.5)

Pick one in the app. All of them end in the HyperReal LUT (strength slider).

- **Look filters only**: the original heavy single-pass look, no upscale.
- **GPU upscale + LUT**: Catmull-Rom 2x upscale with deringing and adaptive sharpen. Meant for a game that renders at half
  the screen size (CODM: 1560x720 on a 3120x1440 screen). Capture runs at half screen size ("Capture at half screen size").
- **Neural upscale + LUT**: the captured frame is cropped to the model size, packed on the GPU, run on the Hexagon NPU through
  LiteRT (int8, planar NCHW, 2x), and the newest result is drawn. Three stages run in parallel (GL, NPU run, NPU read); frames
  are dropped, never queued. The model's input size must match the game's render size, for example 1168x540 for a game that
  renders at 1170x540 (display set to 2340x1080 with `adb shell wm size 1080x2340`).
- **Auto**: neural while at least ~110 results/s are finished, GPU after two slow seconds, back to neural after five fast ones.

Pick the model with "Pick neural model": the `.tflite` from Qualcomm AI Hub, or the `.zip` containing it.
The stats line shows results/s, NPU ms and read ms.

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
