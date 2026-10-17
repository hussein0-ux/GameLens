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
    @Volatile var vignette = 0f       // 0..1    darkened screen corners
    @Volatile var chroma = 0f         // 0..1    chromatic aberration toward the edges
    @Volatile var motion = 0f         // 0..0.6  frame-blend motion blur (blends in the previous frame)
    @Volatile var hdr = 0f            // 0..1    HDR-style local tone mapping + highlight punch (SDR emulation)
    @Volatile var blackPt = 0f        // 0..1    deeper blacks / dehaze


    class Preset(
        val sharpen: Float, val clarity: Float, val contrast: Float, val saturation: Float,
        val filmic: Float, val exposure: Float, val grade: Float, val vibrance: Float, val glow: Float,
        val vignette: Float = 0f, val chroma: Float = 0f, val motion: Float = 0f,
        val hdr: Float = 0f, val blackPt: Float = 0f,
    )

    val NATURAL = Preset(0.50f, 0.25f, 1.03f, 1.05f, 0.20f, 1.05f, 0.10f, 0.20f, 0.05f, hdr = 0.15f, blackPt = 0.10f)
    val VIVID = Preset(0.65f, 0.40f, 1.08f, 1.20f, 0.30f, 1.08f, 0.20f, 0.45f, 0.10f, hdr = 0.35f, blackPt = 0.20f)
    val CINEMATIC = Preset(0.60f, 0.45f, 1.12f, 1.10f, 0.45f, 1.10f, 0.60f, 0.35f, 0.30f, hdr = 0.30f, blackPt = 0.30f)

    /** HDR-style: lifts shadow detail, tames and punches highlights, deep blacks, rich colour. */
    val HDR_POP = Preset(0.65f, 0.45f, 1.10f, 1.18f, 0.25f, 1.06f, 0.15f, 0.50f, 0.15f, hdr = 0.80f, blackPt = 0.35f)

    /** Sharpness first: strong edge sharpening and clarity, no glow, grade or blur. For competitive play. */
    val CRISP = Preset(0.90f, 0.60f, 1.08f, 1.08f, 0.15f, 1.04f, 0.00f, 0.25f, 0.00f, hdr = 0.25f, blackPt = 0.15f)

    /** Edit-style look: heavy teal/orange grade, strong bloom, crisp edges, vignette, fringing, motion blur. */
    val MONTAGE = Preset(0.85f, 0.55f, 1.15f, 1.30f, 0.40f, 1.08f, 0.85f, 0.55f, 0.75f, 0.35f, 0.25f, 0.30f)

    fun apply(p: Preset) {
        sharpen = p.sharpen; clarity = p.clarity; contrast = p.contrast; saturation = p.saturation
        filmic = p.filmic; exposure = p.exposure; grade = p.grade; vibrance = p.vibrance; glow = p.glow
        vignette = p.vignette; chroma = p.chroma; motion = p.motion
        hdr = p.hdr; blackPt = p.blackPt
    }
}
