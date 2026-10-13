# GameLens

A real-time "better graphics" filter for Android gameplay, written in Kotlin + OpenGL ES 3.0.
Built with a Samsung Galaxy S24 Ultra (Android 14+) in mind.

This is **not** NVIDIA DLSS 5. DLSS 5 needs engine data and RTX 50 Tensor Cores. GameLens is a
screen-space post-process: adaptive sharpening, clarity (micro-contrast), a filmic tone curve,
contrast and saturation. It can make footage look crisper and punchier, but it cannot invent
new lighting or materials.

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
3. In the app: press **Start**, grant "Display over other apps", press **Start** again.
4. When Android asks what to share, choose **A single app** and select your game.
5. Switch to the game. To change the look, switch back to GameLens and move the sliders; changes
   apply live. Use the notification's **Stop** button to end.

## Why "single app" matters

In whole-screen capture the overlay would be captured back into itself (a feedback loop that
over-sharpens every frame). Single-app capture only includes the chosen app's window, so the
overlay stays out of the capture.

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
