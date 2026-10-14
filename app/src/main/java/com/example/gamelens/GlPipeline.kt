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

    // Experimental AI upscaler (all of this lives on the GL thread).
    private var upscaler: NpuUpscaler? = null
    private var upscalerPath: String? = null
    private var ai: AiPass? = null
    private var aiFailedPath: String? = null
    private var aiGeneration = 0
    private var lastFrameTs = 0L

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
            if (program != 0) GLES30.glDeleteProgram(program)
            runCatching { ai?.release() }
            runCatching { upscaler?.close() }
            ai = null
            upscaler = null
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

        val ctxAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0)
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
    }

    private fun drawFrame() {
        if (released) return
        surfaceTexture.updateTexImage()
        surfaceTexture.getTransformMatrix(stMatrix)

        val sz = IntArray(1)
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_WIDTH, sz, 0)
        val outW = sz[0]
        EGL14.eglQuerySurface(eglDisplay, eglSurface, EGL14.EGL_HEIGHT, sz, 0)
        val outH = sz[0]

        if (AiSettings.enabled && tryDrawAi(outW, outH)) return

        GLES30.glViewport(0, 0, outW, outH)

        GLES30.glUseProgram(program)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)

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
        EGL14.eglSwapBuffers(eglDisplay, eglSurface)
    }

    /**
     * Experimental path: downscale -> NPU 4x upscale per tile -> present. Returns true if it handled
     * the frame (including "nothing new to draw"); false means the normal look pipeline should run.
     */
    private fun tryDrawAi(outW: Int, outH: Int): Boolean {
        val path = AiSettings.modelPath
        if (path == null) {
            AiSettings.status = "AI: pick an ONNX model in the app first"
            return false
        }
        if (aiGeneration != AiSettings.generation) {
            aiGeneration = AiSettings.generation
            aiFailedPath = null
        }
        if (path == aiFailedPath) return false
        try {
            if (upscaler == null || upscalerPath != path) {
                ai?.release(); ai = null
                upscaler?.close(); upscaler = null
                AiSettings.status = "AI: loading model on the NPU..."
                val up = NpuUpscaler(path, AiSettings.nativeLibDir)
                upscaler = up
                upscalerPath = path
                AiSettings.status = "AI running: " + up.describe()
                Log.i(TAG, AiSettings.status)
            }
            val up = upscaler!!
            val c = AiSettings.cols
            val r = AiSettings.rows
            var pass = ai
            if (pass == null || pass.cols != c || pass.rows != r) {
                pass?.release()
                pass = AiPass(up, c, r).also { it.init() }
                ai = pass
            }
            val ts = surfaceTexture.timestamp
            if (ts == lastFrameTs) return true // no new captured frame since last time
            lastFrameTs = ts
            pass.render(oesTex, stMatrix, quad, outW, outH)
            EGL14.eglSwapBuffers(eglDisplay, eglSurface)
            return true
        } catch (t: Throwable) {
            aiFailedPath = path
            ai?.let { runCatching { it.release() } }
            ai = null
            AiSettings.status = "AI failed, using normal filter: ${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, "AI path failed", t)
            return false
        }
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
    vec3 e = tap(vec2(0.0));
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

    // Clarity: boost mid-frequency detail using a wider 4-tap blur.
    vec3 blur = (tap(vec2(-2.0, 0.0)) + tap(vec2(2.0, 0.0)) + tap(vec2(0.0, -2.0)) + tap(vec2(0.0, 2.0))) * 0.25;
    float midMask = 1.0 - abs(luma(col) * 2.0 - 1.0); // protect pure blacks and whites
    col += (col - blur) * uClarity * (0.5 + 0.5 * midMask);
    col = clamp(col, 0.0, 1.0);

    // Glow: bleed bright areas outward (8 taps) for a soft HDR-style bloom.
    vec3 glowAcc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec3 gs = tap(DIRS[i] * 9.0);
        glowAcc += max(gs - 0.55, 0.0);
    }
    col += (glowAcc / 8.0) * (uGlow * 1.5);

    // Exposure + gentle S-curve: keeps midtones where they were, adds punch.
    col *= uExposure;
    vec3 curved = clamp(col * col * (3.0 - 2.0 * col), 0.0, 1.0);
    col = mix(clamp(col, 0.0, 1.0), curved, uFilmic);
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

    outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""
    }
}
