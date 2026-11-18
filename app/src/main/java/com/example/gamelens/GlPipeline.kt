package com.example.gamelens

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CountDownLatch

/**
 * Takes frames from the screen capture (as an external OES texture), runs them through a single
 * fragment shader and presents the result on [outSurface] (the overlay's SurfaceView).
 *
 * Everything stays on the GPU: capture -> OES texture -> shader -> overlay. No CPU copies.
 */
class GlPipeline(
    private val outSurface: Surface,
    @Volatile private var width: Int,
    @Volatile private var height: Int,
) {
    private val thread = HandlerThread("gl-pipeline").apply { start() }
    private val handler = Handler(thread.looper)

    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE

    private var program = 0
    private var oesTex = 0
    private var uMatrix = 0
    private var uTexel = 0
    private var uSharp = 0
    private var uClarity = 0
    private var uContrast = 0
    private var uSat = 0
    private var uFilmic = 0
    private var uExposure = 0
    private var uGrade = 0
    private var uVibrance = 0
    private var uGlow = 0
    private var uPrev = 0
    private var uRes = 0
    private var uHasPrev = 0
    private var uMotion = 0
    private var uVig = 0
    private var uChroma = 0
    private var uHdr = 0
    private var uBlack = 0
    private var uMicroRelief = 0
    private var uDehaze = 0
    private var uShadowLift = 0
    private var uBypass = 0
    private var uLut = 0
    private var uLutAmt = 0
    private var uLutMap = 0

    // Qualcomm GSR upscaler: capture is first copied (top row first) into a 2D texture, because GSR needs textureGather.
    private var gsrProgram = 0
    private var gsrCopyProgram = 0
    private var gsrCopyMatrix = 0
    private var gsrCopyTexel = 0
    private var gsrCopyAa = 0
    private var gsrPs0 = 0
    private var gsrClarity = 0
    private var gsrVibrance = 0
    private var gsrMicroRelief = 0
    private var gsrDehaze = 0
    private var gsrShadowLift = 0
    private var gsrHdr = 0
    private var gsrFilmic = 0
    private var gsrGlow = 0
    private var gsrExposure = 0
    private var gsrContrast = 0
    private var gsrSat = 0
    private var gsrGrade = 0
    private var gsrBlack = 0
    private var gsrVig = 0
    private var gsrVi = 0
    private var gsrEdge = 0
    private var gsrLut = 0
    private var gsrLutAmt = 0
    private var gsrLutMap = 0
    private val gsrTex = IntArray(1)
    private val gsrFbo = IntArray(1)
    private var gsrW = 0
    private var gsrH = 0

    // Fast upscale program (see UPSCALE_FRAGMENT_SRC).
    private var upProgram = 0
    private var upMatrix = 0
    private var upTexel = 0
    private var upSharp = 0
    private var upClarity = 0
    private var upVibrance = 0
    private var upMicroRelief = 0
    private var upDehaze = 0
    private var upShadowLift = 0
    private var upHdr = 0
    private var upFilmic = 0
    private var upGlow = 0
    private var upExposure = 0
    private var upContrast = 0
    private var upSat = 0
    private var upGrade = 0
    private var upBlack = 0
    private var upVig = 0
    private var upLut = 0
    private var upLutAmt = 0
    private var upLutMap = 0
    private val eglSize = IntArray(1)

    // 3D colour LUT (loaded once from assets; 0 means "not available", the shader step is then skipped).
    private var lutTex = 0
    private var lutSize = 0

    // Motion blur: ping-pong framebuffers hold the previous output frame.
    private var blitProgram = 0
    private var bTex = 0
    private var bRes = 0
    private val fbo = IntArray(1)
    private val fboTex = IntArray(2)
    private var fboW = 0
    private var fboH = 0
    private var cur = 0
    private var hasPrev = false

    lateinit var surfaceTexture: SurfaceTexture
        private set
    lateinit var captureSurface: Surface
        private set

    private val stMatrix = FloatArray(16)
    private val quad: FloatBuffer = ByteBuffer.allocateDirect(16 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            // x, y, u, v
            put(floatArrayOf(
                -1f, -1f, 0f, 0f,
                 1f, -1f, 1f, 0f,
                -1f,  1f, 0f, 1f,
                 1f,  1f, 1f, 1f,
            ))
            position(0)
        }

    @Volatile private var released = false

    // Neural mode (all of this lives on the GL thread).
    private var stage: NeuralStage? = null
    private var neural: NeuralPass? = null
    private var neuralPath: String? = null
    private var lastMode = LookSettings.MODE_GPU
    private var neuralCapW = 0
    private var neuralCapH = 0
    private var infoCache: NeuralModelInfo? = null
    private var infoCachePath: String? = null
    private var neuralFailedPath: String? = null
    private var neuralGeneration = 0
    private var governor = NeuralGovernor()
    private var govSince = System.nanoTime()
    private var govDone = 0
    private var govStart = 0L
    private var neuralFps = 0f
    private var neuProgram = 0
    private var nuPlanar = 0
    private var nuRes = 0
    private var nuDim = 0
    private var nuAff = 0
    private var nuLut = 0
    private var nuLutAmt = 0
    private var nuLutMap = 0

    // Diagnostics, published about once a second.
    private var dgCaptured = 0
    private var dgDrawn = 0
    private var dgSwapNs = 0L
    private var dgSince = System.nanoTime()

    private fun dgPublish() {
        val now = System.nanoTime()
        val dt = (now - dgSince) / 1e9
        if (dt < 1.0) return
        AiSettings.diag = String.format(
            java.util.Locale.US,
            "capture %.0f frames/s, drawn %.0f/s, swap %.1f ms/frame",
            dgCaptured / dt, dgDrawn / dt, if (dgDrawn > 0) dgSwapNs / dgDrawn / 1e6 else 0.0,
        )
        dgCaptured = 0; dgDrawn = 0; dgSwapNs = 0; dgSince = now
    }

    /** Blocks briefly until the GL thread has built the context and the capture surface exists. */
    fun start() {
        val latch = CountDownLatch(1)
        handler.post {
            initEgl()
            initGl()
            latch.countDown()
        }
        latch.await()
    }

    fun resize(w: Int, h: Int) {
        width = w
        height = h
        handler.post {
            if (!released) surfaceTexture.setDefaultBufferSize(w, h)
        }
    }

    fun release() {
        released = true
        handler.post {
            runCatching { surfaceTexture.setOnFrameAvailableListener(null) }
            runCatching { captureSurface.release() }
            runCatching { surfaceTexture.release() }
            if (lutTex != 0) GLES30.glDeleteTextures(1, intArrayOf(lutTex), 0)
            if (program != 0) GLES30.glDeleteProgram(program)
            if (upProgram != 0) GLES30.glDeleteProgram(upProgram)
            if (gsrProgram != 0) GLES30.glDeleteProgram(gsrProgram)
            if (gsrCopyProgram != 0) GLES30.glDeleteProgram(gsrCopyProgram)
            if (gsrTex[0] != 0) GLES30.glDeleteTextures(1, gsrTex, 0)
            if (gsrFbo[0] != 0) GLES30.glDeleteFramebuffers(1, gsrFbo, 0)
            if (blitProgram != 0) GLES30.glDeleteProgram(blitProgram)
            runCatching { neural?.release() }
            runCatching { stage?.close() }
            neural = null
            stage = null
            if (neuProgram != 0) GLES30.glDeleteProgram(neuProgram)
            if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface)
                if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext)
                EGL14.eglTerminate(eglDisplay)
            }
            thread.quitSafely()
        }
    }

    private fun initEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val ver = IntArray(2)
        check(EGL14.eglInitialize(eglDisplay, ver, 0, ver, 1)) { "eglInitialize failed" }

        val cfgAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, 0x0040, // EGL_OPENGL_ES3_BIT_KHR
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val num = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, cfgAttribs, 0, configs, 0, 1, num, 0)
        check(num[0] > 0) { "No EGL config" }

        val ext = EGL14.eglQueryString(eglDisplay, EGL14.EGL_EXTENSIONS) ?: ""
        val highPrioAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
            0x3100, 0x3101, // EGL_CONTEXT_PRIORITY_LEVEL_IMG = EGL_CONTEXT_PRIORITY_HIGH_IMG
            EGL14.EGL_NONE,
        )
        val ctxAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE)
        var ctx = if (ext.contains("EGL_IMG_context_priority")) {
            EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, highPrioAttribs, 0)
        } else {
            EGL14.EGL_NO_CONTEXT
        }
        if (ctx == null || ctx == EGL14.EGL_NO_CONTEXT) {
            ctx = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)
        }
        check(ctx != null && ctx != EGL14.EGL_NO_CONTEXT) { "eglCreateContext failed" }
        eglContext = ctx
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], outSurface, intArrayOf(EGL14.EGL_NONE), 0)
        check(EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) { "eglMakeCurrent failed" }
        EGL14.eglSwapInterval(eglDisplay, 1)
    }

    private fun initGl() {
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        oesTex = tex[0]
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        surfaceTexture = SurfaceTexture(oesTex).apply {
            setDefaultBufferSize(width, height)
            setOnFrameAvailableListener({ drawFrame() }, handler)
        }
        captureSurface = Surface(surfaceTexture)

        program = buildProgram(VERTEX_SRC, FRAGMENT_SRC)
        uMatrix = GLES30.glGetUniformLocation(program, "uStMatrix")
        uTexel = GLES30.glGetUniformLocation(program, "uTexel")
        uSharp = GLES30.glGetUniformLocation(program, "uSharp")
        uClarity = GLES30.glGetUniformLocation(program, "uClarity")
        uContrast = GLES30.glGetUniformLocation(program, "uContrast")
        uSat = GLES30.glGetUniformLocation(program, "uSat")
        uFilmic = GLES30.glGetUniformLocation(program, "uFilmic")
        uExposure = GLES30.glGetUniformLocation(program, "uExposure")
        uGrade = GLES30.glGetUniformLocation(program, "uGrade")
        uVibrance = GLES30.glGetUniformLocation(program, "uVibrance")
        uGlow = GLES30.glGetUniformLocation(program, "uGlow")
        uPrev = GLES30.glGetUniformLocation(program, "uPrev")
        uRes = GLES30.glGetUniformLocation(program, "uRes")
        uHasPrev = GLES30.glGetUniformLocation(program, "uHasPrev")
        uMotion = GLES30.glGetUniformLocation(program, "uMotion")
        uVig = GLES30.glGetUniformLocation(program, "uVig")
        uChroma = GLES30.glGetUniformLocation(program, "uChroma")
        uHdr = GLES30.glGetUniformLocation(program, "uHdr")
        uBlack = GLES30.glGetUniformLocation(program, "uBlack")
        uMicroRelief = GLES30.glGetUniformLocation(program, "uMicroRelief")
        uDehaze = GLES30.glGetUniformLocation(program, "uDehaze")
        uShadowLift = GLES30.glGetUniformLocation(program, "uShadowLift")
        uBypass = GLES30.glGetUniformLocation(program, "uBypass")
        uLut = GLES30.glGetUniformLocation(program, "uLut")
        uLutAmt = GLES30.glGetUniformLocation(program, "uLutAmt")
        uLutMap = GLES30.glGetUniformLocation(program, "uLutMap")
        loadLut()

        initGsr()

        upProgram = buildProgram(VERTEX_SRC, UPSCALE_FRAGMENT_SRC)
        upMatrix = GLES30.glGetUniformLocation(upProgram, "uStMatrix")
        upTexel = GLES30.glGetUniformLocation(upProgram, "uTexel")
        upSharp = GLES30.glGetUniformLocation(upProgram, "uSharp")
        upClarity = GLES30.glGetUniformLocation(upProgram, "uClarity")
        upVibrance = GLES30.glGetUniformLocation(upProgram, "uVibrance")
        upMicroRelief = GLES30.glGetUniformLocation(upProgram, "uMicroRelief")
        upDehaze = GLES30.glGetUniformLocation(upProgram, "uDehaze")
        upShadowLift = GLES30.glGetUniformLocation(upProgram, "uShadowLift")
        upHdr = GLES30.glGetUniformLocation(upProgram, "uHdr")
        upFilmic = GLES30.glGetUniformLocation(upProgram, "uFilmic")
        upGlow = GLES30.glGetUniformLocation(upProgram, "uGlow")
        upExposure = GLES30.glGetUniformLocation(upProgram, "uExposure")
        upContrast = GLES30.glGetUniformLocation(upProgram, "uContrast")
        upSat = GLES30.glGetUniformLocation(upProgram, "uSat")
        upGrade = GLES30.glGetUniformLocation(upProgram, "uGrade")
        upBlack = GLES30.glGetUniformLocation(upProgram, "uBlack")
        upVig = GLES30.glGetUniformLocation(upProgram, "uVig")
        upLut = GLES30.glGetUniformLocation(upProgram, "uLut")
        upLutAmt = GLES30.glGetUniformLocation(upProgram, "uLutAmt")
        upLutMap = GLES30.glGetUniformLocation(upProgram, "uLutMap")

        neuProgram = buildProgram(BLIT_VERTEX_SRC, NEURAL_FRAGMENT_SRC)
        nuPlanar = GLES30.glGetUniformLocation(neuProgram, "uPlanar")
        nuRes = GLES30.glGetUniformLocation(neuProgram, "uRes")
        nuDim = GLES30.glGetUniformLocation(neuProgram, "uDim")
        nuAff = GLES30.glGetUniformLocation(neuProgram, "uAff")
        nuLut = GLES30.glGetUniformLocation(neuProgram, "uLut")
        nuLutAmt = GLES30.glGetUniformLocation(neuProgram, "uLutAmt")
        nuLutMap = GLES30.glGetUniformLocation(neuProgram, "uLutMap")

        blitProgram = buildProgram(BLIT_VERTEX_SRC, BLIT_FRAGMENT_SRC)
        bTex = GLES30.glGetUniformLocation(blitProgram, "uTex2")
        bRes = GLES30.glGetUniformLocation(blitProgram, "uRes")
    }

    /** Reads the .cube from assets into a 3D texture. Any failure leaves the LUT off instead of breaking the frame. */
    private fun loadLut() {
        try {
            val text = GameLensApp.instance.assets.open(LUT_ASSET).bufferedReader().use { it.readText() }
            val lut = parseCube(text)
            val data = ByteBuffer.allocateDirect(lut.rgb.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            data.put(lut.rgb).position(0)

            val tex = IntArray(1)
            GLES30.glGenTextures(1, tex, 0)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, tex[0])
            GLES30.glTexImage3D(
                GLES30.GL_TEXTURE_3D, 0, GLES30.GL_RGB16F, lut.size, lut.size, lut.size, 0,
                GLES30.GL_RGB, GLES30.GL_FLOAT, data,
            )
            for (axis in intArrayOf(GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_TEXTURE_WRAP_R)) {
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, axis, GLES30.GL_CLAMP_TO_EDGE)
            }
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            check(GLES30.glGetError() == GLES30.GL_NO_ERROR) { "GL error while uploading the LUT" }
            lutTex = tex[0]
            lutSize = lut.size
            Log.i(TAG, "LUT loaded: ${lut.size}^3")
        } catch (t: Throwable) {
            lutTex = 0
            Log.e(TAG, "LUT not available, colour grade step skipped", t)
        }
    }

    private fun ensureFbo(w: Int, h: Int) {
        if (fboW == w && fboH == h && fboTex[0] != 0) return
        if (fboTex[0] != 0) GLES30.glDeleteTextures(2, fboTex, 0)
        if (fbo[0] == 0) GLES30.glGenFramebuffers(1, fbo, 0)
        GLES30.glGenTextures(2, fboTex, 0)
        for (i in 0..1) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTex[i])
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        }
        fboW = w; fboH = h; cur = 0; hasPrev = false
    }

    private var drawErrors = 0

    /** Never lets an exception escape the GL thread (that would kill the whole app). */
    private fun drawFrame() {
        if (released) return
        try {
            drawFrameUnsafe()
        } catch (t: Throwable) {
            if (released) return
            Log.e(TAG, "drawFrame failed", t)
            if (++drawErrors <= 3) runCatching { GameLensApp.record(GameLensApp.instance, "drawFrame", t) }
            AiSettings.status = "Draw error (skipped frame): ${t.javaClass.simpleName}: ${t.message}"
        }
    }

    private fun drawFrameUnsafe() {
        dgCaptured++
        surfaceTexture.updateTexImage()
        surfaceTexture.getTransformMatrix(stMatrix)

        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_WIDTH, eglSize, 0)
        val outW = eglSize[0]
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_HEIGHT, eglSize, 0)
        val outH = eglSize[0]

        val mode = LookSettings.mode
        if (mode < LookSettings.MODE_NEURAL && (neural != null || lastMode >= LookSettings.MODE_NEURAL)) {
            // Left neural mode: stop the NPU work and clear the old neural stats so they are not mistaken for live numbers.
            releaseNeural()
            AiSettings.status = "Neural idle (not selected)"
        }
        lastMode = mode
        if (!LookSettings.bypass && mode >= LookSettings.MODE_NEURAL && drawNeural(outW, outH, mode)) return
        if (!LookSettings.bypass && mode >= LookSettings.MODE_GPU) {
            if (LookSettings.upscaler == LookSettings.UPSCALER_GSR && gsrProgram != 0) drawGsr(outW, outH)
            else drawUpscale(outW, outH)
            return
        }

        val useMotion = LookSettings.motion > 0.01f
        if (useMotion) {
            ensureFbo(outW, outH)
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[0])
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, fboTex[cur], 0)
        } else {
            hasPrev = false
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        }
        GLES30.glViewport(0, 0, outW, outH)

        GLES30.glUseProgram(program)
        // Unit 1 always holds a 2D texture for uPrev (the previous output frame while motion blur is on).
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (fboTex[0] != 0) fboTex[1 - cur] else 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glUniform1i(uPrev, 1)
        // The LUT sampler must have its own unit: two sampler types may not share unit 0 with the external texture.
        GLES30.glUniform1i(uLut, 2)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, lutTex)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glUniform1f(uBypass, if (LookSettings.bypass) 1f else 0f)
        GLES30.glUniform1f(uLutAmt, if (lutTex != 0) LookSettings.lut else 0f)
        if (lutSize > 0) {
            // Map colour 0..1 onto texel centres so the first and last LUT entries are hit exactly.
            GLES30.glUniform2f(uLutMap, (lutSize - 1f) / lutSize, 0.5f / lutSize)
        }
        GLES30.glUniform2f(uRes, outW.toFloat(), outH.toFloat())
        GLES30.glUniform1f(uHasPrev, if (useMotion && hasPrev) 1f else 0f)
        GLES30.glUniform1f(uMotion, if (useMotion) LookSettings.motion else 0f)
        GLES30.glUniform1f(uVig, LookSettings.vignette)
        GLES30.glUniform1f(uChroma, LookSettings.chroma)
        GLES30.glUniform1f(uHdr, LookSettings.hdr)
        GLES30.glUniform1f(uBlack, LookSettings.blackPt)
        GLES30.glUniform1f(uMicroRelief, LookSettings.microRelief)
        GLES30.glUniform1f(uDehaze, LookSettings.dehaze)
        GLES30.glUniform1f(uShadowLift, LookSettings.shadowLift)

        GLES30.glUniformMatrix4fv(uMatrix, 1, false, stMatrix, 0)
        GLES30.glUniform2f(uTexel, 1f / width, 1f / height)
        GLES30.glUniform1f(uSharp, LookSettings.sharpen)
        GLES30.glUniform1f(uClarity, LookSettings.clarity)
        GLES30.glUniform1f(uContrast, LookSettings.contrast)
        GLES30.glUniform1f(uSat, LookSettings.saturation)
        GLES30.glUniform1f(uFilmic, LookSettings.filmic)
        GLES30.glUniform1f(uExposure, LookSettings.exposure)
        GLES30.glUniform1f(uGrade, LookSettings.grade)
        GLES30.glUniform1f(uVibrance, LookSettings.vibrance)
        GLES30.glUniform1f(uGlow, LookSettings.glow)

        quad.position(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(0)
        quad.position(2)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(1)

        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

        if (useMotion) {
            // Copy the finished frame to the screen; it also becomes "previous" for the next frame.
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, outW, outH)
            GLES30.glUseProgram(blitProgram)
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTex[cur])
            GLES30.glUniform1i(bTex, 0)
            GLES30.glUniform2f(bRes, outW.toFloat(), outH.toFloat())
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            cur = 1 - cur
            hasPrev = true
        }
        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
        // The capture/drawn counters were only published by the AI path; count the plain look path too.
        dgDrawn++
        dgPublish()
    }

    /** Fast path: one pass from the captured frame straight to the screen (see UPSCALE_FRAGMENT_SRC). */
    private fun drawUpscale(outW: Int, outH: Int) {
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, outW, outH)
        GLES30.glUseProgram(upProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glUniform1i(upLut, 2)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, lutTex)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glUniform1f(upLutAmt, if (lutTex != 0) LookSettings.lut else 0f)
        if (lutSize > 0) GLES30.glUniform2f(upLutMap, (lutSize - 1f) / lutSize, 0.5f / lutSize)
        GLES30.glUniformMatrix4fv(upMatrix, 1, false, stMatrix, 0)
        GLES30.glUniform2f(upTexel, 1f / width, 1f / height)
        GLES30.glUniform1f(upSharp, LookSettings.upSharp)
        GLES30.glUniform1f(upClarity, LookSettings.upClarity)
        GLES30.glUniform1f(upVibrance, LookSettings.upVibrance)
        GLES30.glUniform1f(upMicroRelief, LookSettings.microRelief)
        GLES30.glUniform1f(upDehaze, LookSettings.dehaze)
        GLES30.glUniform1f(upShadowLift, LookSettings.shadowLift)
        GLES30.glUniform1f(upHdr, LookSettings.hdr)
        GLES30.glUniform1f(upFilmic, LookSettings.filmic)
        GLES30.glUniform1f(upGlow, LookSettings.glow)
        GLES30.glUniform1f(upExposure, LookSettings.exposure)
        GLES30.glUniform1f(upContrast, LookSettings.contrast)
        GLES30.glUniform1f(upSat, LookSettings.saturation)
        GLES30.glUniform1f(upGrade, LookSettings.grade)
        GLES30.glUniform1f(upBlack, LookSettings.blackPt)
        GLES30.glUniform1f(upVig, LookSettings.vignette)

        drawQuad()

        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
        dgDrawn++
        dgPublish()
    }

    /** Needs OpenGL ES 3.1 (textureGather). On any problem GSR stays off and the Catmull-Rom upscaler is used. */
    private fun initGsr() {
        try {
            val version = GLES30.glGetString(GLES30.GL_VERSION) ?: ""
            val m = Regex("OpenGL ES (\\d+)\\.(\\d+)").find(version)
            val ok = m != null && (m.groupValues[1].toInt() > 3 || (m.groupValues[1].toInt() == 3 && m.groupValues[2].toInt() >= 1))
            check(ok) { "needs OpenGL ES 3.1, found '$version'" }
            val frag = GameLensApp.instance.assets.open("sgsr1_mobile.frag").bufferedReader().use { it.readText() }
            gsrProgram = buildProgram(GSR_VERTEX_SRC, frag)
            gsrCopyProgram = buildProgram(GSR_COPY_VERTEX_SRC, GSR_COPY_FRAGMENT_SRC)
            gsrCopyMatrix = GLES30.glGetUniformLocation(gsrCopyProgram, "uStMatrix")
            gsrCopyTexel = GLES30.glGetUniformLocation(gsrCopyProgram, "uTexel")
            gsrCopyAa = GLES30.glGetUniformLocation(gsrCopyProgram, "uAa")
            gsrPs0 = GLES30.glGetUniformLocation(gsrProgram, "ps0")
            gsrVi = GLES30.glGetUniformLocation(gsrProgram, "ViewportInfo")
            gsrEdge = GLES30.glGetUniformLocation(gsrProgram, "uEdgeSharpness")
            gsrLut = GLES30.glGetUniformLocation(gsrProgram, "uLut")
            gsrLutAmt = GLES30.glGetUniformLocation(gsrProgram, "uLutAmt")
            gsrLutMap = GLES30.glGetUniformLocation(gsrProgram, "uLutMap")
            gsrClarity = GLES30.glGetUniformLocation(gsrProgram, "uClarity")
            gsrVibrance = GLES30.glGetUniformLocation(gsrProgram, "uVibrance")
            gsrMicroRelief = GLES30.glGetUniformLocation(gsrProgram, "uMicroRelief")
            gsrDehaze = GLES30.glGetUniformLocation(gsrProgram, "uDehaze")
            gsrShadowLift = GLES30.glGetUniformLocation(gsrProgram, "uShadowLift")
            gsrHdr = GLES30.glGetUniformLocation(gsrProgram, "uHdr")
            gsrFilmic = GLES30.glGetUniformLocation(gsrProgram, "uFilmic")
            gsrGlow = GLES30.glGetUniformLocation(gsrProgram, "uGlow")
            gsrExposure = GLES30.glGetUniformLocation(gsrProgram, "uExposure")
            gsrContrast = GLES30.glGetUniformLocation(gsrProgram, "uContrast")
            gsrSat = GLES30.glGetUniformLocation(gsrProgram, "uSat")
            gsrGrade = GLES30.glGetUniformLocation(gsrProgram, "uGrade")
            gsrBlack = GLES30.glGetUniformLocation(gsrProgram, "uBlack")
            gsrVig = GLES30.glGetUniformLocation(gsrProgram, "uVig")
        } catch (t: Throwable) {
            gsrProgram = 0
            gsrCopyProgram = 0
            Log.w(TAG, "GSR unavailable, using Catmull-Rom: ${t.message}")
            AiSettings.status = "GSR unavailable (${t.message}), using Catmull-Rom"
        }
    }

    private fun ensureGsrTarget() {
        if (gsrW == width && gsrH == height && gsrTex[0] != 0) return
        if (gsrTex[0] != 0) GLES30.glDeleteTextures(1, gsrTex, 0)
        if (gsrFbo[0] != 0) GLES30.glDeleteFramebuffers(1, gsrFbo, 0)
        gsrW = width
        gsrH = height
        GLES30.glGenTextures(1, gsrTex, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gsrTex[0])
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, gsrW, gsrH, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glGenFramebuffers(1, gsrFbo, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, gsrFbo[0])
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, gsrTex[0], 0)
        check(GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) == GLES30.GL_FRAMEBUFFER_COMPLETE) { "GSR framebuffer incomplete" }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    /** GSR path: capture -> 2D texture (top row first) -> Qualcomm GSR 2x -> HyperReal LUT -> screen. */
    private fun drawGsr(outW: Int, outH: Int) {
        ensureGsrTarget()
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, gsrFbo[0])
        GLES30.glViewport(0, 0, gsrW, gsrH)
        GLES30.glUseProgram(gsrCopyProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glUniformMatrix4fv(gsrCopyMatrix, 1, false, stMatrix, 0)
        GLES30.glUniform2f(gsrCopyTexel, 1f / gsrW, 1f / gsrH)
        GLES30.glUniform1f(gsrCopyAa, LookSettings.aa)
        drawQuad()

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, outW, outH)
        GLES30.glUseProgram(gsrProgram)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, gsrTex[0])
        GLES30.glUniform1i(gsrPs0, 0)
        GLES30.glUniform4f(gsrVi, 1f / gsrW, 1f / gsrH, gsrW.toFloat(), gsrH.toFloat())
        GLES30.glUniform1f(gsrEdge, 1f + LookSettings.upSharp)
        GLES30.glUniform1f(gsrClarity, LookSettings.upClarity)
        GLES30.glUniform1f(gsrVibrance, LookSettings.upVibrance)
        GLES30.glUniform1f(gsrMicroRelief, LookSettings.microRelief)
        GLES30.glUniform1f(gsrDehaze, LookSettings.dehaze)
        GLES30.glUniform1f(gsrShadowLift, LookSettings.shadowLift)
        GLES30.glUniform1f(gsrHdr, LookSettings.hdr)
        GLES30.glUniform1f(gsrFilmic, LookSettings.filmic)
        GLES30.glUniform1f(gsrGlow, LookSettings.glow)
        GLES30.glUniform1f(gsrExposure, LookSettings.exposure)
        GLES30.glUniform1f(gsrContrast, LookSettings.contrast)
        GLES30.glUniform1f(gsrSat, LookSettings.saturation)
        GLES30.glUniform1f(gsrGrade, LookSettings.grade)
        GLES30.glUniform1f(gsrBlack, LookSettings.blackPt)
        GLES30.glUniform1f(gsrVig, LookSettings.vignette)
        GLES30.glUniform1i(gsrLut, 2)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, lutTex)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glUniform1f(gsrLutAmt, if (lutTex != 0) LookSettings.lut else 0f)
        if (lutSize > 0) GLES30.glUniform2f(gsrLutMap, (lutSize - 1f) / lutSize, 0.5f / lutSize)
        drawQuad()
        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
        dgDrawn++
        dgPublish()
    }

    private fun drawQuad() {
        quad.position(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(0)
        quad.position(2)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
    }

    /**
     * Neural path. Every captured frame is cropped, packed and offered to the NPU; the newest finished result is shown.
     * Returns true if it drew the frame. False means "draw the GPU picture instead" (no result yet, NPU failed,
     * wrong capture size, or Auto mode has fallen back).
     */
    private fun drawNeural(outW: Int, outH: Int, mode: Int): Boolean {
        val n = ensureNeural() ?: return false
        try {
            n.process(oesTex, stMatrix)
            n.uploadLatest()
        } catch (t: Throwable) {
            failNeural("Neural pass failed: ${t.javaClass.simpleName}: ${t.message}", t)
            return false
        }
        stage?.error?.let { failNeural("NPU failed: ${it.javaClass.simpleName}: ${it.message}", it); return false }
        tickGovernor(mode)
        if (!n.hasResult) return false
        if (mode == LookSettings.MODE_AUTO && !governor.useNeural) return false

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, outW, outH)
        GLES30.glUseProgram(neuProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, n.planarTex[0])
        GLES30.glUniform1i(nuPlanar, 1)
        GLES30.glUniform1i(nuLut, 2)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, lutTex)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glUniform2f(nuRes, outW.toFloat(), outH.toFloat())
        GLES30.glUniform2f(nuDim, n.outW.toFloat(), n.outH.toFloat())
        // Dequantise: value = (q - zero point) * scale, where q = signed byte + 128.
        GLES30.glUniform2f(nuAff, 255f * n.info.outScale, -n.info.outZp * n.info.outScale)
        GLES30.glUniform1f(nuLutAmt, if (lutTex != 0) LookSettings.lut else 0f)
        if (lutSize > 0) GLES30.glUniform2f(nuLutMap, (lutSize - 1f) / lutSize, 0.5f / lutSize)
        drawQuad()
        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
        dgDrawn++
        dgPublish()
        return true
    }

    /** Creates the NPU stage and GL pass on first use (or when a new model is picked). Null = not usable, see status. */
    private fun ensureNeural(): NeuralPass? {
        val path = AiSettings.modelPath
        if (path == null) {
            AiSettings.status = "Neural: pick a .tflite model in the app first"
            return null
        }
        if (neuralGeneration != AiSettings.generation) {
            neuralGeneration = AiSettings.generation
            neuralFailedPath = null
            infoCache = null
        }
        if (path == neuralFailedPath) return null
        val have = neural
        if (have != null && neuralPath == path && neuralCapW == width && neuralCapH == height) return have
        if (have != null) releaseNeural()
        try {
            val file = java.io.File(path)
            val info = infoCache?.takeIf { infoCachePath == path } ?: readNeuralModelInfo(file).also {
                infoCache = it
                infoCachePath = path
            }
            // A size mismatch is not a failure: the capture is portrait until the game turns the screen, and the
            // display may be changed later. Wait quietly and check again on the next frame.
            if (width - info.w !in 0..8 || height - info.h !in 0..8) {
                AiSettings.status = "Neural waiting: capture is ${width}x$height, the model wants ${info.w}x${info.h}. " +
                    "Set the display so the game renders at that size (landscape) and capture at half size. Showing GPU."
                return null
            }
            AiSettings.status = "Neural: loading model on the NPU..."
            val st = NeuralStage(GameLensApp.instance, file, info.w / 4 * info.h * 3 * 4)
            stage = st
            val pass = NeuralPass(st, info, width, height, this::buildProgram, this::drawQuad)
            pass.init()
            neural = pass
            neuralPath = path
            neuralCapW = width
            neuralCapH = height
            governor = NeuralGovernor()
            govSince = System.nanoTime()
            govStart = govSince
            govDone = 0
            neuralFps = 0f
            Log.i(TAG, "Neural ready: ${info.w}x${info.h} x${info.scale}")
            return pass
        } catch (t: Throwable) {
            failNeural("Neural unavailable, using GPU upscale: ${t.javaClass.simpleName}: ${t.message}", t)
            return null
        }
    }

    private fun releaseNeural() {
        runCatching { neural?.release() }
        runCatching { stage?.close() }
        neural = null
        stage = null
        neuralPath = null
    }

    private fun failNeural(message: String, t: Throwable) {
        neuralFailedPath = AiSettings.modelPath
        releaseNeural()
        AiSettings.status = message
        Log.e(TAG, message, t)
    }

    /** Once a second: measure how many neural frames finished, feed the Auto governor, update the status text. */
    private fun tickGovernor(mode: Int) {
        val now = System.nanoTime()
        val dt = (now - govSince) / 1e9f
        if (dt < 1f) return
        val st = stage ?: return
        val done = st.finishedCount()
        neuralFps = (done - govDone) / dt
        govDone = done
        govSince = now
        if (now - govStart > 3_000_000_000L) governor.onSecond(neuralFps) // ignore the first 3 s (warm-up)
        val showing = if (mode == LookSettings.MODE_AUTO && !governor.useNeural) "GPU (Auto fell back)" else "neural"
        AiSettings.status = String.format(
            java.util.Locale.US,
            "Neural %.0f results/s, NPU %.1f ms, read %.1f ms, showing %s",
            neuralFps, st.runMs, st.readMs, showing,
        )
    }

    private fun buildProgram(vs: String, fs: String): Int {
        fun compile(type: Int, src: String): Int {
            val s = GLES30.glCreateShader(type)
            GLES30.glShaderSource(s, src)
            GLES30.glCompileShader(s)
            val ok = IntArray(1)
            GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
            check(ok[0] != 0) { "Shader compile error: " + GLES30.glGetShaderInfoLog(s) }
            return s
        }
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vs))
        GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fs))
        GLES30.glLinkProgram(p)
        val ok = IntArray(1)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "Program link error: " + GLES30.glGetProgramInfoLog(p) }
        return p
    }

    companion object {
        private const val TAG = "GameLensAI"
        private const val LUT_ASSET = "hyperreal_rec709_33.cube"
        private const val VERTEX_SRC = """#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform mat4 uStMatrix;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUv = (uStMatrix * vec4(aUv, 0.0, 1.0)).xy;
}
"""

        // Single pass: CAS-style adaptive sharpening -> clarity (wide unsharp) -> filmic curve,
        // contrast, saturation. All tunable live from the app UI.
        // GSR: texture coordinates start at the TOP left (the copy pass stores the picture top row first).
        private const val GSR_VERTEX_SRC = """#version 310 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out highp vec4 in_TEXCOORD0;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    in_TEXCOORD0 = vec4(aUv.x, 1.0 - aUv.y, 0.0, 0.0);
}
"""
        // Copy pass: draws the capture upside down on purpose, so row 0 of the 2D texture is the top of the picture.
        private const val GSR_COPY_VERTEX_SRC = """#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform mat4 uStMatrix;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos.x, -aPos.y, 0.0, 1.0);
    vUv = (uStMatrix * vec4(aUv, 0.0, 1.0)).xy;
}
"""
        private const val GSR_COPY_FRAGMENT_SRC = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES uTex;
uniform vec2 uTexel;   // one capture texel in uv units
uniform float uAa;     // 0 = plain copy
in vec2 vUv;
out vec4 outColor;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
// Edge anti-aliasing (FXAA 3 console variant). Only high-contrast pixels are touched, so texture detail is kept.
void main() {
    vec3 m = texture(uTex, vUv).rgb;
    if (uAa <= 0.0) { outColor = vec4(m, 1.0); return; }
    float nw = luma(texture(uTex, vUv + vec2(-1.0, -1.0) * uTexel).rgb);
    float ne = luma(texture(uTex, vUv + vec2( 1.0, -1.0) * uTexel).rgb);
    float sw = luma(texture(uTex, vUv + vec2(-1.0,  1.0) * uTexel).rgb);
    float se = luma(texture(uTex, vUv + vec2( 1.0,  1.0) * uTexel).rgb);
    float lm = luma(m);
    float lumaMin = min(lm, min(min(nw, ne), min(sw, se)));
    float lumaMax = max(lm, max(max(nw, ne), max(sw, se)));
    if (lumaMax - lumaMin < max(0.0312, lumaMax * 0.125)) { outColor = vec4(m, 1.0); return; }
    vec2 dir = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    float dirReduce = max((nw + ne + sw + se) * 0.25 * 0.125, 1.0 / 128.0);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * uTexel;
    vec3 a = 0.5 * (texture(uTex, vUv + dir * (1.0 / 3.0 - 0.5)).rgb + texture(uTex, vUv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(uTex, vUv - dir * 0.5).rgb + texture(uTex, vUv + dir * 0.5).rgb);
    float lb = luma(b);
    vec3 aa = (lb < lumaMin || lb > lumaMax) ? a : b;
    outColor = vec4(mix(m, aa, uAa), 1.0);
}
"""

        private const val BLIT_VERTEX_SRC = """#version 300 es
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
"""

        private const val BLIT_FRAGMENT_SRC = """#version 300 es
precision highp float;
uniform sampler2D uTex2;
uniform vec2 uRes;
out vec4 outColor;
void main() { outColor = vec4(texture(uTex2, gl_FragCoord.xy / uRes).rgb, 1.0); }
"""

        // Fast path: Catmull-Rom 2x upscale (9 bilinear fetches), deringing clamp, CAS sharpen + fused HyperReal pass + LUT.
        private const val UPSCALE_FRAGMENT_SRC = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;

uniform samplerExternalOES uTex;
uniform vec2 uTexel;
uniform float uSharp;
uniform float uClarity;
uniform float uVibrance;
uniform float uMicroRelief;
uniform float uDehaze;
uniform float uShadowLift;
uniform float uHdr;
uniform float uFilmic;
uniform float uGlow;
uniform float uExposure;
uniform float uContrast;
uniform float uSat;
uniform float uGrade;
uniform float uBlack;
uniform float uVig;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

in vec2 vUv;
out vec4 outColor;

vec3 tex(vec2 p) { return texture(uTex, p).rgb; }
float luma709(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec2 size = 1.0 / uTexel;
    vec2 pos = vUv * size;
    vec2 p1 = floor(pos - 0.5) + 0.5;
    vec2 f = pos - p1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 p0 = (p1 - 1.0) * uTexel;
    vec2 p3 = (p1 + 2.0) * uTexel;
    vec2 p12 = (p1 + w2 / w12) * uTexel;

    vec3 c = tex(vec2(p0.x, p0.y)) * w0.x * w0.y
           + tex(vec2(p12.x, p0.y)) * w12.x * w0.y
           + tex(vec2(p3.x, p0.y)) * w3.x * w0.y
           + tex(vec2(p0.x, p12.y)) * w0.x * w12.y
           + tex(vec2(p12.x, p12.y)) * w12.x * w12.y
           + tex(vec2(p3.x, p12.y)) * w3.x * w12.y
           + tex(vec2(p0.x, p3.y)) * w0.x * w3.y
           + tex(vec2(p12.x, p3.y)) * w12.x * w3.y
           + tex(vec2(p3.x, p3.y)) * w3.x * w3.y;

    // Neighbours one output pixel away (half a source texel at 2x): used for deringing, sharpening, and micro-relief.
    vec2 d = 0.5 * uTexel;
    vec3 n = tex(vUv + vec2(0.0, -d.y));
    vec3 s = tex(vUv + vec2(0.0,  d.y));
    vec3 w = tex(vUv + vec2(-d.x, 0.0));
    vec3 e = tex(vUv + vec2( d.x, 0.0));
    vec3 m = tex(vUv);
    vec3 mn = min(m, min(min(n, s), min(w, e)));
    vec3 mx = max(m, max(max(n, s), max(w, e)));
    c = clamp(c, mn, mx);

    // Contrast-adaptive sharpen (same idea as CAS): less where the local range is already high.
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    vec3 nearAvg = (n + s + w + e) * 0.25;
    vec3 col = clamp(c + (c - nearAvg) * uSharp * amp * 2.0, 0.0, 1.0);

    if ((uClarity + uMicroRelief + uDehaze + uShadowLift + uHdr + uGlow) > 0.001) {
        vec2 o2 = uTexel * 2.75;
        vec3 tNW = tex(vUv + vec2(-o2.x, -o2.y));
        vec3 tNE = tex(vUv + vec2( o2.x, -o2.y));
        vec3 tSW = tex(vUv + vec2(-o2.x,  o2.y));
        vec3 tSE = tex(vUv + vec2( o2.x,  o2.y));
        vec3 wideAvg = (tNW + tNE + tSW + tSE) * 0.25;

        if (uDehaze > 0.001) {
            vec3 darkRgb = min(col, min(min(tNW, tNE), min(tSW, tSE)));
            float darkCh = min(darkRgb.r, min(darkRgb.g, darkRgb.b));
            float veil = clamp(darkCh * 0.42 * uDehaze, 0.0, 0.45);
            col = clamp((col - veil) / max(1.0 - veil, 0.55), 0.0, 1.0);
        }
        if (uMicroRelief > 0.001) {
            vec3 hpFine = col - nearAvg;
            vec3 hpMed = nearAvg - wideAvg;
            float lc0 = luma709(col);
            float reliefMask = clamp(4.0 * lc0 * (1.0 - lc0), 0.15, 1.0);
            col = clamp(col + (hpFine * 1.15 + hpMed * 0.55) * (uMicroRelief * reliefMask), 0.0, 1.0);
        }
        if (uClarity > 0.001) {
            float lc1 = luma709(col);
            float midMask = 1.0 - abs(lc1 * 2.0 - 1.0);
            col = clamp(col + vec3((lc1 - luma709(wideAvg)) * uClarity * (0.5 + 0.5 * midMask)), 0.0, 1.0);
        }
        float localL = luma709(wideAvg);
        if (uShadowLift > 0.001) {
            float lc2 = luma709(col);
            float shMask = (1.0 - smoothstep(0.0, 0.45, lc2)) * (1.0 - smoothstep(0.05, 0.55, localL));
            col = clamp(col + col * (shMask * uShadowLift * 0.90) + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
        }
        if (uGlow > 0.001) {
            vec3 bloom = max(wideAvg - 0.58, vec3(0.0)) + max(nearAvg - 0.65, vec3(0.0)) * 0.5;
            col = col + bloom * (uGlow * 1.25);
        }
        if (uHdr > 0.001) {
            float gain = clamp(sqrt(clamp(0.42 / (localL + 0.04), 0.0, 4.0)), 0.90, 1.65);
            col *= mix(1.0, gain, uHdr * 0.55);
            float lc3 = luma709(col);
            col += col * smoothstep(0.60, 1.0, lc3) * uHdr * 0.28;
            col = mix(col, col / (1.0 + max(col - 0.85, vec3(0.0)) * 1.2), uHdr);
        }
    }

    col *= uExposure;
    if (uFilmic > 0.001) col = mix(clamp(col, 0.0, 1.0), acesFilm(max(col, vec3(0.0))), uFilmic);
    col = (col - 0.5) * uContrast + 0.5;
    if (uGrade > 0.001) {
        float l0 = luma709(col);
        col += uGrade * (vec3(-0.03, 0.03, 0.08) * (1.0 - smoothstep(0.0, 0.55, l0))
                       + vec3( 0.08, 0.03,-0.05) * smoothstep(0.45, 1.0, l0));
    }
    if (uVibrance > 0.001) {
        float satNow = max(col.x, max(col.y, col.z)) - min(col.x, min(col.y, col.z));
        float lv = luma709(col);
        col = vec3(lv) + (col - vec3(lv)) * (1.0 + uVibrance * (1.0 - satNow));
    }
    col = mix(vec3(luma709(col)), col, uSat);
    if (uBlack > 0.001) {
        float bp = uBlack * 0.07;
        col = max(col - bp, vec3(0.0)) / (1.0 - bp);
    }
    if (uVig > 0.001) col *= 1.0 - uVig * smoothstep(0.30, 0.80, length(vUv - 0.5));
    col = clamp(col, 0.0, 1.0);
    if (uLutAmt > 0.0) col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
    outColor = vec4(col, 1.0);
}
"""

        // Neural picture: planar signed bytes -> colour -> HyperReal LUT, drawn full screen.
        private const val NEURAL_FRAGMENT_SRC = """#version 300 es
precision highp float;
uniform highp sampler2D uPlanar;
uniform vec2 uRes;
uniform vec2 uDim;
uniform vec2 uAff;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;
out vec4 outColor;

// p: 0..1 over the picture, y = 0 is the TOP row. The three colour planes are stacked in one texture.
vec3 fetch(vec2 p) {
    float y = clamp(p.y, 0.5 / uDim.y, 1.0 - 0.5 / uDim.y);
    float r = textureLod(uPlanar, vec2(p.x, y / 3.0), 0.0).r;
    float g = textureLod(uPlanar, vec2(p.x, (y + 1.0) / 3.0), 0.0).r;
    float b = textureLod(uPlanar, vec2(p.x, (y + 2.0) / 3.0), 0.0).r;
    vec3 v = (vec3(r, g, b) * 127.0 + 128.0) / 255.0;
    return clamp(v * uAff.x + uAff.y, 0.0, 1.0);
}

void main() {
    vec2 p = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
    vec3 col = fetch(p);
    if (uLutAmt > 0.0) col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
    outColor = vec4(col, 1.0);
}
"""

        private const val FRAGMENT_SRC = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;

uniform samplerExternalOES uTex;
uniform vec2 uTexel;
uniform float uSharp;
uniform float uClarity;
uniform float uContrast;
uniform float uSat;
uniform float uFilmic;
uniform float uExposure;
uniform float uGrade;
uniform float uVibrance;
uniform float uGlow;
uniform sampler2D uPrev;
uniform vec2 uRes;
uniform float uHasPrev;
uniform float uMotion;
uniform float uVig;
uniform float uChroma;
uniform float uHdr;
uniform float uBlack;
uniform float uMicroRelief;
uniform float uDehaze;
uniform float uShadowLift;
uniform float uBypass;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

const vec2 DIRS[8] = vec2[8](
    vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0), vec2(-0.7071, 0.7071),
    vec2(-1.0, 0.0), vec2(-0.7071, -0.7071), vec2(0.0, -1.0), vec2(0.7071, -0.7071)
);

in vec2 vUv;
out vec4 outColor;

vec3 tap(vec2 off) { return texture(uTex, vUv + off * uTexel).rgb; }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    if (uBypass > 0.5) {
        outColor = vec4(texture(uTex, vUv).rgb, 1.0);   // one texture fetch: shows the raw capture rate
        return;
    }
    vec2 suv = gl_FragCoord.xy / uRes;
    // Chromatic fringe: red and blue are sampled slightly apart, more toward the screen edges.
    vec2 cd = (suv - 0.5) * uChroma * 8.0;
    vec3 e = vec3(tap(cd).r, tap(vec2(0.0)).g, tap(-cd).b);
    vec3 n = tap(vec2(0.0, -1.0));
    vec3 s = tap(vec2(0.0,  1.0));
    vec3 w = tap(vec2(-1.0, 0.0));
    vec3 ea = tap(vec2( 1.0, 0.0));

    // Contrast-adaptive sharpening: back off where the local range is already high.
    vec3 mn = min(e, min(min(n, s), min(w, ea)));
    vec3 mx = max(e, max(max(n, s), max(w, ea)));
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    float peak = mix(-0.125, -0.2, uSharp);
    vec3 wgt = amp * peak;
    vec3 sharp = (e + (n + s + w + ea) * wgt) / (1.0 + 4.0 * wgt);
    vec3 col = mix(e, sharp, step(0.001, uSharp));

    vec3 nearAvg = (n + s + w + ea) * 0.25;
    vec3 blur = (tap(vec2(-2.0, 0.0)) + tap(vec2(2.0, 0.0)) + tap(vec2(0.0, -2.0)) + tap(vec2(0.0, 2.0))) * 0.25;

    // Dark-channel atmospheric dehaze
    if (uDehaze > 0.001) {
        vec3 nwD = tap(vec2(-3.0, -3.0));
        vec3 seD = tap(vec2( 3.0,  3.0));
        vec3 dRgb = min(col, min(nwD, seD));
        float veil = clamp(min(dRgb.r, min(dRgb.g, dRgb.b)) * 0.42 * uDehaze, 0.0, 0.45);
        col = clamp((col - veil) / max(1.0 - veil, 0.55), 0.0, 1.0);
    }

    // Dual-band micro-relief
    if (uMicroRelief > 0.001) {
        vec3 hpFine = col - nearAvg;
        vec3 hpMed = nearAvg - blur;
        float lcR = luma(col);
        float rMask = clamp(4.0 * lcR * (1.0 - lcR), 0.15, 1.0);
        col = clamp(col + (hpFine * 1.15 + hpMed * 0.55) * (uMicroRelief * rMask), 0.0, 1.0);
    }

    // Clarity: boost mid-frequency detail using a wider 4-tap blur.
    float midMask = 1.0 - abs(luma(col) * 2.0 - 1.0); // protect pure blacks and whites
    col += (col - blur) * uClarity * (0.5 + 0.5 * midMask);
    col = clamp(col, 0.0, 1.0);

    // Tactical shadow detail recovery
    if (uShadowLift > 0.001) {
        float lcS = luma(col);
        float locS = luma(blur);
        float shMask = (1.0 - smoothstep(0.0, 0.45, lcS)) * (1.0 - smoothstep(0.05, 0.55, locS));
        col = clamp(col + col * (shMask * uShadowLift * 0.90) + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
    }

    // Glow: bleed bright areas outward (8 taps) for a soft HDR-style bloom.
    vec3 glowAcc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec3 gs = tap(DIRS[i] * 9.0);
        glowAcc += max(gs - 0.55, 0.0);
    }
    col += (glowAcc / 8.0) * (uGlow * 1.5);
    // Strong glow settings add a second, wider ring so the bloom spreads like an edit's glow effect.
    float wideW = smoothstep(0.4, 1.0, uGlow);
    if (wideW > 0.0) {
        vec3 wideAcc = vec3(0.0);
        for (int i = 0; i < 8; i++) {
            wideAcc += max(tap(DIRS[i] * 24.0) - 0.6, vec3(0.0));
        }
        col += (wideAcc / 8.0) * (wideW * 1.2);
    }

    // HDR-style look (SDR emulation): local tone mapping, then highlight punch.
    //  - the neighbourhood brightness (wide taps) decides a gain: dark areas are lifted, bright areas eased,
    //    so shadow and highlight detail both stay visible;
    //  - highlights are then pushed toward the top of the range with their colour preserved.
    if (uHdr > 0.001) {
        vec3 wide = (tap(vec2(-14.0, -14.0)) + tap(vec2(14.0, -14.0)) + tap(vec2(-14.0, 14.0)) + tap(vec2(14.0, 14.0))
                   + tap(vec2(-26.0, 0.0)) + tap(vec2(26.0, 0.0)) + tap(vec2(0.0, -26.0)) + tap(vec2(0.0, 26.0))) * 0.125;
        float localL = luma(wide);
        float gain = clamp(pow(0.42 / (localL + 0.04), 0.5), 0.92, 1.7);
        col *= mix(1.0, gain, uHdr * 0.55);
        float lc = luma(col);
        float hi = smoothstep(0.62, 1.0, lc);
        col += col * hi * uHdr * 0.30;
        // Roll the very top off smoothly instead of hard-clipping, so bright areas keep detail.
        col = mix(col, col / (1.0 + max(col - 0.85, 0.0) * 1.2), uHdr);
    }

    // Exposure + ACES Filmic curve + contrast.
    col *= uExposure;
    col = mix(clamp(col, 0.0, 1.0), aces(max(col, vec3(0.0))), uFilmic);
    col = (col - 0.5) * uContrast + 0.5;

    // Split-tone grade: cool teal in the shadows, warm orange in the highlights.
    float l0 = luma(col);
    float shadowsW = 1.0 - smoothstep(0.0, 0.55, l0);
    float highsW = smoothstep(0.45, 1.0, l0);
    col += uGrade * (vec3(-0.03, 0.03, 0.08) * shadowsW + vec3(0.08, 0.03, -0.05) * highsW);

    // Vibrance (lifts muted colours more than already-vivid ones), then saturation.
    float chromaMax = max(col.r, max(col.g, col.b));
    float chromaMin = min(col.r, min(col.g, col.b));
    float satNow = chromaMax - chromaMin;
    col = mix(vec3(luma(col)), col, 1.0 + uVibrance * (1.0 - satNow));
    col = mix(vec3(luma(col)), col, uSat);

    // Deeper blacks: lowers the black point (dehaze) and re-expands, keeping whites where they were.
    float bp = uBlack * 0.07;
    col = max(col - bp, 0.0) / (1.0 - bp);

    col *= 1.0 - uVig * smoothstep(0.30, 0.80, length(suv - 0.5));
    col = clamp(col, 0.0, 1.0);
    if (uLutAmt > 0.0) col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
    if (uMotion > 0.0 && uHasPrev > 0.5) col = mix(col, texture(uPrev, suv).rgb, uMotion);
    outColor = vec4(col, 1.0);
}
"""
    }
}
