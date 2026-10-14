package com.example.gamelens

/** Shared state for the experimental AI upscaler. UI writes it, the GL thread reads it every frame. */
object AiSettings {
    @Volatile var enabled = false
    @Volatile var modelPath: String? = null
    /** Tile grid: the model input is cols*128 x rows*128 pixels, the output is 4x that. */
    @Volatile var cols = 4
    @Volatile var rows = 2
    /** 0 = AI only, 1 = split (AI left, plain bilinear right), 2 = bilinear only. */
    @Volatile var viewMode = 0
    /** Bumped by the UI to make the pipeline retry after a failure. */
    @Volatile var generation = 0
    @Volatile var nativeLibDir = ""
    @Volatile var status = "AI idle"
    @Volatile var stats = ""
}
