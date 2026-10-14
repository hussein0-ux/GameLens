package com.example.gamelens

/** Live-tunable look settings. Written by the UI thread, read by the GL thread every frame. */
object LookSettings {
    @Volatile var sharpen = 0.65f     // 0..1    adaptive (CAS-style) sharpening
    @Volatile var clarity = 0.40f     // 0..1    local contrast / micro-contrast
    @Volatile var contrast = 1.08f    // 0.8..1.4
    @Volatile var saturation = 1.20f  // 0.5..1.6
    @Volatile var filmic = 0.30f      // 0..1    blend toward a gentle S-curve
    @Volatile var exposure = 1.08f    // 0.8..1.4 brightness (compensates the S-curve)
    @Volatile var grade = 0.20f       // 0..1    teal shadows / warm orange highlights
    @Volatile var vibrance = 0.45f    // 0..1    boosts muted colours more than vivid ones
    @Volatile var glow = 0.20f        // 0..1    bloom around bright areas

    class Preset(
        val sharpen: Float, val clarity: Float, val contrast: Float, val saturation: Float,
        val filmic: Float, val exposure: Float, val grade: Float, val vibrance: Float, val glow: Float,
    )

    val NATURAL = Preset(0.50f, 0.25f, 1.03f, 1.05f, 0.20f, 1.05f, 0.10f, 0.20f, 0.10f)
    val VIVID = Preset(0.65f, 0.40f, 1.08f, 1.20f, 0.30f, 1.08f, 0.20f, 0.45f, 0.20f)
    val CINEMATIC = Preset(0.60f, 0.45f, 1.12f, 1.10f, 0.45f, 1.10f, 0.60f, 0.35f, 0.35f)

    fun apply(p: Preset) {
        sharpen = p.sharpen; clarity = p.clarity; contrast = p.contrast; saturation = p.saturation
        filmic = p.filmic; exposure = p.exposure; grade = p.grade; vibrance = p.vibrance; glow = p.glow
    }
}
