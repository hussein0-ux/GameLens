package com.example.gamelens

import kotlin.test.Test
import kotlin.test.assertEquals

class NeuralGovernorTest {
    private fun feed(g: NeuralGovernor, vararg fps: Float) = fps.forEach { g.onSecond(it) }

    @Test fun startsOnNeuralAndStaysWhileFast() {
        val g = NeuralGovernor()
        feed(g, 119f, 120f, 118f, 119f)
        assertEquals(true, g.useNeural)
    }

    @Test fun oneSlowSecondDoesNotSwitch() {
        val g = NeuralGovernor()
        feed(g, 100f, 119f, 100f, 119f)
        assertEquals(true, g.useNeural)
    }

    @Test fun twoSlowSecondsInARowSwitchToGpu() {
        val g = NeuralGovernor()
        feed(g, 100f, 105f)
        assertEquals(false, g.useNeural)
    }

    @Test fun returnsToNeuralOnlyAfterFiveFastSeconds() {
        val g = NeuralGovernor()
        feed(g, 90f, 90f)
        feed(g, 120f, 120f, 120f, 120f)
        assertEquals(false, g.useNeural)
        feed(g, 120f)
        assertEquals(true, g.useNeural)
    }

    @Test fun aSlowSecondResetsRecovery() {
        val g = NeuralGovernor()
        feed(g, 90f, 90f)
        feed(g, 120f, 120f, 120f, 120f, 110f, 120f, 120f, 120f, 120f)
        assertEquals(false, g.useNeural)
        feed(g, 120f)
        assertEquals(true, g.useNeural)
    }

    @Test fun afterReturningItNeedsTwoNewSlowSecondsToLeaveAgain() {
        val g = NeuralGovernor()
        feed(g, 90f, 90f, 120f, 120f, 120f, 120f, 120f)
        assertEquals(true, g.useNeural)
        feed(g, 100f)
        assertEquals(true, g.useNeural)
        feed(g, 100f)
        assertEquals(false, g.useNeural)
    }
}
