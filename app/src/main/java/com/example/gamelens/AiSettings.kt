package com.example.gamelens

/** Shared state for the neural upscaler. UI writes it, the GL thread reads it every frame. */
object AiSettings {
    @Volatile var modelPath: String? = null
    /** Bumped by the UI to make the pipeline retry after a failure. */
    @Volatile var generation = 0
    @Volatile var nativeLibDir = ""
    @Volatile var status = "Neural idle"
    @Volatile var stats = ""
    /** Capture/swap diagnostics from the GL thread. */
    @Volatile var diag = ""
}
