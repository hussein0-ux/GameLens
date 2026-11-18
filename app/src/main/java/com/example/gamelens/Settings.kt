package com.example.gamelens

/** Live-tunable look settings. Written by the UI thread, read by the GL thread every frame. */
object LookSettings {
    const val MODE_LOOK = 0    // the original heavy look filters, no upscale
    const val MODE_GPU = 1     // GPU 2x upscale (GSR / Catmull-Rom) + fused HyperReal pass + LUT
    const val MODE_NEURAL = 2  // NPU 2x upscale + LUT (falls back to GPU for frames the NPU has not finished)
    const val MODE_AUTO = 3    // neural while it holds the frame rate, GPU otherwise

    const val UPSCALER_CATMULL = 0  // Catmull-Rom + adaptive sharpen (GameLens shader)
    const val UPSCALER_GSR = 1      // Qualcomm Snapdragon Game Super Resolution 1 (edge-directed)

    @Volatile var sharpen = 0.92f     // 0..1    adaptive (CAS-style) sharpening
    @Volatile var clarity = 0.68f     // 0..1    local contrast / micro-contrast
    @Volatile var contrast = 1.14f    // 0.8..1.4
    @Volatile var saturation = 1.16f  // 0.5..1.6
    @Volatile var filmic = 0.52f      // 0..1    blend toward ACES Filmic curve
    @Volatile var exposure = 1.06f    // 0.8..1.4 brightness (compensates tone mapping)
    @Volatile var grade = 0.22f       // 0..1    teal shadows / warm orange highlights
    @Volatile var vibrance = 0.45f    // 0..1    boosts muted colours more than vivid ones
    @Volatile var glow = 0.22f        // 0..1    micro-bloom around bright specular highlights
    @Volatile var vignette = 0.08f    // 0..1    subtle darkened screen corners
    @Volatile var chroma = 0f         // 0..1    chromatic aberration toward the edges (MODE_LOOK)
    @Volatile var motion = 0f         // 0..0.6  frame-blend motion blur (MODE_LOOK)
    @Volatile var hdr = 0.85f         // 0..1    HDR-style local tone mapping + highlight punch (SDR emulation)
    @Volatile var blackPt = 0.24f     // 0..1    deeper blacks
    @Volatile var microRelief = 0.72f // 0..1    dual-band micro-texture high-pass relief (Adreno 750 fused)
    @Volatile var dehaze = 0.38f      // 0..1    dark-channel atmospheric haze removal
    @Volatile var shadowLift = 0.42f  // 0..1    tactical shadow detail recovery for dark corners
    @Volatile var lut = 1.0f          // 0..1    strength of the 3D colour LUT (HyperReal Rec709)
    @Volatile var mode = MODE_GPU     // MODE_LOOK, MODE_GPU, MODE_NEURAL or MODE_AUTO
    @Volatile var upscaler = UPSCALER_GSR // which GPU upscaler MODE_GPU uses
    @Volatile var upSharp = 0.92f     // 0..1    sharpening strength of the upscale pass
    @Volatile var aa = 0.65f          // 0..1    edge anti-aliasing before the GSR upscale (0 = off)
    @Volatile var upClarity = 0.68f   // 0..1    local contrast after the GPU upscale (0 = off)
    @Volatile var upVibrance = 0.45f  // 0..1    boosts muted colours after the GPU upscale (0 = off)
    @Volatile var captureHalf = true  // capture at half the screen size (CoD Mobile render res); read at start
    @Volatile var target120Hz = true  // hint SurfaceFlinger to lock overlay Surface to 120 Hz (applies on start)
    @Volatile var bypass = false      // true: copy the captured frame straight to the screen (capture-rate test)

    class Preset(
        val sharpen: Float, val clarity: Float, val contrast: Float, val saturation: Float,
        val filmic: Float, val exposure: Float, val grade: Float, val vibrance: Float, val glow: Float,
        val vignette: Float = 0f, val chroma: Float = 0f, val motion: Float = 0f,
        val hdr: Float = 0f, val blackPt: Float = 0f,
        val microRelief: Float = 0f, val dehaze: Float = 0f, val shadowLift: Float = 0f,
    )

    val NATURAL = Preset(
        0.50f, 0.25f, 1.03f, 1.05f, 0.20f, 1.05f, 0.10f, 0.20f, 0.05f,
        hdr = 0.15f, blackPt = 0.10f, microRelief = 0.20f, dehaze = 0.10f, shadowLift = 0.15f,
    )
    val VIVID = Preset(
        0.65f, 0.40f, 1.08f, 1.20f, 0.30f, 1.08f, 0.20f, 0.45f, 0.10f,
        hdr = 0.35f, blackPt = 0.20f, microRelief = 0.35f, dehaze = 0.20f, shadowLift = 0.20f,
    )
    val CINEMATIC = Preset(
        0.60f, 0.45f, 1.12f, 1.10f, 0.45f, 1.10f, 0.60f, 0.35f, 0.30f,
        hdr = 0.30f, blackPt = 0.30f, microRelief = 0.35f, dehaze = 0.25f, shadowLift = 0.25f,
    )

    /** HDR-style: lifts shadow detail, tames and punches highlights, deep blacks, rich colour. */
    val HDR_POP = Preset(
        0.65f, 0.45f, 1.10f, 1.18f, 0.25f, 1.06f, 0.15f, 0.50f, 0.15f,
        hdr = 0.80f, blackPt = 0.35f, microRelief = 0.45f, dehaze = 0.25f, shadowLift = 0.40f,
    )

    /** Sharpness first: strong edge sharpening and clarity, no glow, grade or blur. For competitive play. */
    val CRISP = Preset(
        0.90f, 0.60f, 1.08f, 1.08f, 0.15f, 1.04f, 0.00f, 0.25f, 0.00f,
        hdr = 0.25f, blackPt = 0.15f, microRelief = 0.55f, dehaze = 0.25f, shadowLift = 0.30f,
    )

    /** Edit-style look: heavy teal/orange grade, strong bloom, crisp edges, vignette, fringing, motion blur. */
    val MONTAGE = Preset(
        0.85f, 0.55f, 1.15f, 1.30f, 0.40f, 1.08f, 0.85f, 0.55f, 0.75f,
        vignette = 0.35f, chroma = 0.25f, motion = 0.30f,
        hdr = 0.45f, blackPt = 0.25f, microRelief = 0.50f, dehaze = 0.20f, shadowLift = 0.15f,
    )

    /**
     * Snapdragon 8 Gen 3 (Adreno 750) Extreme HyperReal 120 Hz profile tuned for Call of Duty: Mobile.
     * Combines Qualcomm GSR edge upscale + dual-band micro-relief + dark-channel dehaze +
     * tactical shadow visibility + local HDR tone mapping + ACES Filmic + HyperReal 3D LUT in one pass.
     */
    val SD8_GEN3_HYPER = Preset(
        sharpen = 0.92f,
        clarity = 0.68f,
        contrast = 1.14f,
        saturation = 1.16f,
        filmic = 0.52f,
        exposure = 1.06f,
        grade = 0.22f,
        vibrance = 0.45f,
        glow = 0.22f,
        vignette = 0.08f,
        chroma = 0f,
        motion = 0f,
        hdr = 0.85f,
        blackPt = 0.24f,
        microRelief = 0.72f,
        dehaze = 0.38f,
        shadowLift = 0.42f,
    )

    fun apply(p: Preset) {
        sharpen = p.sharpen; clarity = p.clarity; contrast = p.contrast; saturation = p.saturation
        filmic = p.filmic; exposure = p.exposure; grade = p.grade; vibrance = p.vibrance; glow = p.glow
        vignette = p.vignette; chroma = p.chroma; motion = p.motion
        hdr = p.hdr; blackPt = p.blackPt
        microRelief = p.microRelief; dehaze = p.dehaze; shadowLift = p.shadowLift
        upClarity = p.clarity
        upVibrance = p.vibrance
    }

    /** Configures the full zero-stall 120 Hz Snapdragon 8 Gen 3 pipeline (GPU GSR + half capture + Extreme HyperReal). */
    fun applySd8Gen3Hyper120() {
        apply(SD8_GEN3_HYPER)
        mode = MODE_GPU
        upscaler = UPSCALER_GSR
        captureHalf = true
        target120Hz = true
        bypass = false
        upSharp = 0.92f
        aa = 0.65f
        lut = 1.0f
    }
}
