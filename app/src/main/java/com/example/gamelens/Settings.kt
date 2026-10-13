package com.example.gamelens

/** Live-tunable look settings. Written by the UI thread, read by the GL thread every frame. */
object LookSettings {
    @Volatile var sharpen = 0.6f      // 0..1  adaptive (CAS-style) sharpening
    @Volatile var clarity = 0.35f     // 0..1  local contrast / micro-contrast
    @Volatile var contrast = 1.05f    // 0.8..1.4
    @Volatile var saturation = 1.10f  // 0.5..1.6
    @Volatile var filmic = 0.35f      // 0..1  blend toward a filmic highlight/shadow curve
}
