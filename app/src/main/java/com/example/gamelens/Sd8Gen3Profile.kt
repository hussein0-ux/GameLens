package com.example.gamelens

/**
 * Snapdragon 8 Gen 3 (Adreno 750 @ 903 MHz) 120 Hz performance & image-quality budget model.
 * Kept pure Kotlin (zero Android dependencies) so unit tests can verify 120 fps safety directly.
 */
object Sd8Gen3Profile {
    const val REFRESH_HZ = 120f
    const val FRAME_BUDGET_MS = 1000f / REFRESH_HZ // 8.33 ms
    /** Target upper bound for the GameLens overlay pass so CoD Mobile + SurfaceFlinger stay locked at 120 fps. */
    const val OVERLAY_TARGET_MS = 4.50f

    data class FrameEstimate(
        val gpuMs: Float,
        val safe120Hz: Boolean,
        val headroomMs: Float,
    )

    fun estimate(
        mode: Int = LookSettings.mode,
        upscaler: Int = LookSettings.upscaler,
        captureHalf: Boolean = LookSettings.captureHalf,
        aa: Float = LookSettings.aa,
        clarity: Float = LookSettings.upClarity,
        microRelief: Float = LookSettings.microRelief,
        dehaze: Float = LookSettings.dehaze,
        shadowLift: Float = LookSettings.shadowLift,
        hdr: Float = LookSettings.hdr,
        glow: Float = LookSettings.glow,
        motion: Float = LookSettings.motion,
    ): FrameEstimate {
        val baseMs = when {
            mode == LookSettings.MODE_LOOK -> 3.25f
            upscaler == LookSettings.UPSCALER_GSR -> 1.35f
            else -> 1.10f
        }
        val resScale = if (captureHalf) 1.0f else 2.55f
        val aaCost = if (mode != LookSettings.MODE_LOOK && upscaler == LookSettings.UPSCALER_GSR && aa > 0.01f) 0.35f else 0f
        val hyperCost = if (clarity > 0.01f || microRelief > 0.01f || dehaze > 0.01f || shadowLift > 0.01f || hdr > 0.01f) {
            if (mode == LookSettings.MODE_LOOK) 1.15f else 0.55f
        } else 0f
        val glowCost = if (glow > 0.01f) {
            if (mode == LookSettings.MODE_LOOK) 0.90f else 0.15f
        } else 0f
        val motionCost = if (mode == LookSettings.MODE_LOOK && motion > 0.01f) 0.70f else 0f

        val total = (baseMs + aaCost + hyperCost + glowCost + motionCost) * resScale
        val rounded = (total * 100f).toInt() / 100f
        return FrameEstimate(
            gpuMs = rounded,
            safe120Hz = total <= OVERLAY_TARGET_MS,
            headroomMs = ((FRAME_BUDGET_MS - total) * 100f).toInt() / 100f,
        )
    }

    /** Exact clamp bounds matching the GLSL shaders so UI values never produce NaNs or blown channels. */
    fun sanitizeUnit(v: Float): Float = v.coerceIn(0f, 1f)
}
