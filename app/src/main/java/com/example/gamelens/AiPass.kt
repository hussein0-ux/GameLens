package com.example.gamelens

import android.opengl.GLES30
import java.nio.FloatBuffer

class AiPassOptimized(private val up: NpuUpscaler, val cols: Int, val rows: Int) {
    
    // Load your newly optimized native pipeline runtime library
    init {
        System.loadLibrary("native_npu_pipeline")
    }

    private external fun nativeInitBuffers(w: Int, h: Int)
    private external fun nativeProcessFrame(textureId: Int, matrix: FloatArray)
    private external fun nativeRelease()

    fun init() {
        // Initialize the native zero-copy EGL structures and AHardwareBuffer bindings
        nativeInitBuffers(cols * up.inW, rows * up.inH)
    }

    fun release() {
        nativeRelease()
    }

    fun render(oesTex: Int, stMatrix: FloatArray, quad: FloatBuffer, screenW: Int, screenH: Int) {
        // Delete the original rgbaBuf allocation, glReadPixels, fillTile loops, and glTexSubImage2D!
        // The complete frame state is passed directly to the native graphics engine.
        nativeProcessFrame(oesTex, stMatrix)
    }
}
