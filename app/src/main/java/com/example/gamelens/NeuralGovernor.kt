package com.example.gamelens

/**
 * Decides, once per second, whether Auto mode shows the neural picture or the GPU picture.
 * Leaves neural after [downSeconds] slow seconds in a row; returns only after [upSeconds] fast seconds in a row,
 * so it does not flicker between the two looks.
 */
class NeuralGovernor(
    private val minFps: Float = 110f,
    private val downSeconds: Int = 2,
    private val upFps: Float = 118f,
    private val upSeconds: Int = 5,
) {
    var useNeural = true
        private set
    private var slow = 0
    private var fast = 0

    /** [neuralFps] is how many neural frames finished during the last second. */
    fun onSecond(neuralFps: Float) {
        if (useNeural) {
            slow = if (neuralFps < minFps) slow + 1 else 0
            if (slow >= downSeconds) { useNeural = false; fast = 0 }
        } else {
            fast = if (neuralFps >= upFps) fast + 1 else 0
            if (fast >= upSeconds) { useNeural = true; slow = 0 }
        }
    }
}
