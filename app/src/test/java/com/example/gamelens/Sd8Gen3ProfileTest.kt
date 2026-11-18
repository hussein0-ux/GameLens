package com.example.gamelens

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class Sd8Gen3ProfileTest {

    @Test
    fun extremePresetHolds120HzBudgetUnderHalfCaptureGsr() {
        LookSettings.applySd8Gen3Hyper120()
        val est = Sd8Gen3Profile.estimate()
        assertEquals(LookSettings.MODE_GPU, LookSettings.mode)
        assertEquals(LookSettings.UPSCALER_GSR, LookSettings.upscaler)
        assertTrue(LookSettings.captureHalf)
        assertTrue(LookSettings.target120Hz)
        assertTrue(est.safe120Hz, "Expected <= ${Sd8Gen3Profile.OVERLAY_TARGET_MS} ms, got ${est.gpuMs} ms")
        assertTrue(est.headroomMs > 3.0f, "Expected comfortable 120Hz headroom, got ${est.headroomMs} ms")
    }

    @Test
    fun fullResHeavyLookExceeds120HzTargetBudget() {
        val heavy = Sd8Gen3Profile.estimate(
            mode = LookSettings.MODE_LOOK,
            captureHalf = false,
            clarity = 0.8f,
            microRelief = 0.8f,
            dehaze = 0.5f,
            shadowLift = 0.5f,
            hdr = 0.9f,
            glow = 0.7f,
            motion = 0.3f,
        )
        assertFalse(heavy.safe120Hz)
        assertTrue(heavy.gpuMs > Sd8Gen3Profile.FRAME_BUDGET_MS)
    }

    @Test
    fun sgsrShaderDeclaresAllFusedHyperRealUniforms() {
        val shader = File("src/main/assets/sgsr1_mobile.frag").readText()
        val required = listOf(
            "uEdgeSharpness", "ViewportInfo", "ps0",
            "uClarity", "uVibrance", "uMicroRelief", "uDehaze", "uShadowLift",
            "uHdr", "uFilmic", "uGlow", "uExposure", "uContrast", "uSat",
            "uGrade", "uBlack", "uVig", "uLut", "uLutAmt", "uLutMap",
        )
        for (u in required) {
            assertTrue(shader.contains(u), "Missing uniform '$u' in sgsr1_mobile.frag")
        }
    }

    @Test
    fun applyPresetSyncsGpuClarityAndVibrance() {
        LookSettings.apply(LookSettings.HDR_POP)
        assertEquals(LookSettings.HDR_POP.clarity, LookSettings.upClarity)
        assertEquals(LookSettings.HDR_POP.vibrance, LookSettings.upVibrance)
        assertEquals(LookSettings.HDR_POP.microRelief, LookSettings.microRelief)
        assertEquals(LookSettings.HDR_POP.dehaze, LookSettings.dehaze)
        assertEquals(LookSettings.HDR_POP.shadowLift, LookSettings.shadowLift)
    }
}
