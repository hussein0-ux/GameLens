package com.example.gamelens

import android.opengl.GLES11Ext
import android.opengl.GLES30
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale

internal fun buildGlProgram(vs: String, fs: String): Int {
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

/**
 * One AI frame:
 *  1. GL: box-filter the captured screen down to (cols*128) x (rows*128) into an FBO.
 *  2. CPU: read it back, cut it into 128x128 tiles, run each tile through the NPU (4x upscale).
 *  3. GL: upload each float output tile into one big atlas texture.
 *  4. GL: draw the atlas to the overlay with bilinear filtering (optionally split against plain bilinear).
 * Must be created, used and released on the GL thread.
 */
class AiPass(private val up: NpuUpscaler, val cols: Int, val rows: Int) {

    private val lowW = cols * up.inW
    private val lowH = rows * up.inH
    private val fullW = cols * up.outW
    private val fullH = rows * up.outH

    private var fbo = 0
    private var lowTex = 0
    private var atlas = 0
    private var copyProg = 0
    private var presentProg = 0
    private var uCopyMatrix = 0
    private var uCopyFoot = 0
    private var uAtlas = 0
    private var uLow = 0
    private var uTileOut = 0
    private var uFull = 0
    private var uPlanar = 0
    private var uMode = 0
    private var uQuant = 0
    private var uOutScale = 0
    private var uOutZero = 0

    private val rgbaBuf: ByteBuffer = ByteBuffer.allocateDirect(lowW * lowH * 4).order(ByteOrder.nativeOrder())
    private val rgba = ByteArray(lowW * lowH * 4)
    private val tileFloats = FloatArray(3 * up.inW * up.inH)
    private val tileIn: FloatBuffer =
        ByteBuffer.allocateDirect(tileFloats.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private var scratch: FloatBuffer? = null

    // uint8 (w8a8) models: raw pixel bytes in, quantised bytes out. Dequantised in the present shader.
    private val q = up.quantized
    private val tileBytes = ByteArray(if (q) 3 * up.inW * up.inH else 0)
    private val tileInU8: ByteBuffer =
        ByteBuffer.allocateDirect(tileBytes.size).order(ByteOrder.nativeOrder())
    private var scratchU8: ByteBuffer? = null

    // Rolling stats, published every STATS_EVERY frames.
    private var frames = 0
    private var sumRead = 0L
    private var sumConv = 0L
    private var sumNpu = 0L
    private var sumUp = 0L
    private var sumTotal = 0L
    private var lastPublish = System.nanoTime()

    fun init() {
        val maxTex = IntArray(1)
        GLES30.glGetIntegerv(GLES30.GL_MAX_TEXTURE_SIZE, maxTex, 0)
        val atlasH = if (up.planarOut) fullH * 3 else fullH
        check(fullW <= maxTex[0] && atlasH <= maxTex[0]) {
            "Atlas ${fullW}x$atlasH exceeds GL max texture size ${maxTex[0]}; pick a smaller grid"
        }

        val t = IntArray(1)
        // Low-res render target (also sampled for the plain-bilinear comparison).
        GLES30.glGenTextures(1, t, 0); lowTex = t[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, lowTex)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, lowW, lowH, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glGenFramebuffers(1, t, 0); fbo = t[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, lowTex, 0)
        check(GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) == GLES30.GL_FRAMEBUFFER_COMPLETE) { "FBO incomplete" }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        // Atlas of float output tiles. Nearest filtering is required for texelFetch on float textures.
        GLES30.glGenTextures(1, t, 0); atlas = t[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, atlas)
        // uint8 models keep the atlas as 8-bit (4x less upload and memory); the shader dequantises.
        if (q) {
            if (up.planarOut) {
                GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8, fullW, atlasH, 0, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, null)
            } else {
                GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGB8, fullW, atlasH, 0, GLES30.GL_RGB, GLES30.GL_UNSIGNED_BYTE, null)
            }
        } else if (up.planarOut) {
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R32F, fullW, atlasH, 0, GLES30.GL_RED, GLES30.GL_FLOAT, null)
        } else {
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGB32F, fullW, atlasH, 0, GLES30.GL_RGB, GLES30.GL_FLOAT, null)
        }
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        copyProg = buildGlProgram(COPY_VS, COPY_FS)
        uCopyMatrix = GLES30.glGetUniformLocation(copyProg, "uStMatrix")
        uCopyFoot = GLES30.glGetUniformLocation(copyProg, "uFootprint")
        presentProg = buildGlProgram(PRESENT_VS, PRESENT_FS)
        uAtlas = GLES30.glGetUniformLocation(presentProg, "uAtlas")
        uLow = GLES30.glGetUniformLocation(presentProg, "uLow")
        uTileOut = GLES30.glGetUniformLocation(presentProg, "uTileOut")
        uFull = GLES30.glGetUniformLocation(presentProg, "uFull")
        uPlanar = GLES30.glGetUniformLocation(presentProg, "uPlanar")
        uMode = GLES30.glGetUniformLocation(presentProg, "uMode")
        uQuant = GLES30.glGetUniformLocation(presentProg, "uQuant")
        uOutScale = GLES30.glGetUniformLocation(presentProg, "uOutScale")
        uOutZero = GLES30.glGetUniformLocation(presentProg, "uOutZero")
    }

    fun release() {
        if (copyProg != 0) GLES30.glDeleteProgram(copyProg)
        if (presentProg != 0) GLES30.glDeleteProgram(presentProg)
        GLES30.glDeleteTextures(2, intArrayOf(lowTex, atlas), 0)
        GLES30.glDeleteFramebuffers(1, intArrayOf(fbo), 0)
        copyProg = 0; presentProg = 0
    }

    private fun bindQuad(quad: FloatBuffer) {
        quad.position(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(0)
        quad.position(2)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, 16, quad)
        GLES30.glEnableVertexAttribArray(1)
    }

    fun render(oesTex: Int, stMatrix: FloatArray, quad: FloatBuffer, screenW: Int, screenH: Int) {
        val t0 = System.nanoTime()

        // 1. Downscale the captured screen into the low-res FBO.
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, lowW, lowH)
        GLES30.glUseProgram(copyProg)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glUniformMatrix4fv(uCopyMatrix, 1, false, stMatrix, 0)
        GLES30.glUniform2f(uCopyFoot, 1f / lowW, 1f / lowH)
        bindQuad(quad)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)

        // 2. Read it back (this waits for the GPU, so it includes the downscale time).
        rgbaBuf.clear()
        GLES30.glReadPixels(0, 0, lowW, lowH, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, rgbaBuf)
        rgbaBuf.rewind()
        rgbaBuf.get(rgba)
        val t1 = System.nanoTime()
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        // 3. Tiles through the NPU, straight into the atlas.
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, atlas)
        var conv = 0L
        var npu = 0L
        var upl = 0L
        for (ty in 0 until rows) for (tx in 0 until cols) {
            val c0 = System.nanoTime()
            fillTile(tx, ty)
            if (q) {
                tileInU8.clear()
                tileInU8.put(tileBytes)
                tileInU8.rewind()
            } else {
                tileIn.clear()
                tileIn.put(tileFloats)
                tileIn.rewind()
            }
            val c1 = System.nanoTime()
            conv += c1 - c0
            var cbStart = 0L
            var cbEnd = 0L
            if (q) {
                up.inferU8(tileInU8) { out ->
                    cbStart = System.nanoTime()
                    uploadU8(out, tx, ty)
                    cbEnd = System.nanoTime()
                }
            } else {
                up.infer(tileIn) { out ->
                    cbStart = System.nanoTime()
                    upload(out, tx, ty)
                    cbEnd = System.nanoTime()
                }
            }
            val c2 = System.nanoTime()
            npu += (cbStart - c1) + (c2 - cbEnd)
            upl += cbEnd - cbStart
        }
        val t2 = System.nanoTime()

        // 4. Present.
        GLES30.glViewport(0, 0, screenW, screenH)
        GLES30.glUseProgram(presentProg)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, atlas)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, lowTex)
        GLES30.glUniform1i(uAtlas, 0)
        GLES30.glUniform1i(uLow, 1)
        GLES30.glUniform2i(uTileOut, up.outW, up.outH)
        GLES30.glUniform2i(uFull, fullW, fullH)
        GLES30.glUniform1i(uPlanar, if (up.planarOut) 1 else 0)
        GLES30.glUniform1i(uMode, AiSettings.viewMode)
        GLES30.glUniform1i(uQuant, if (q) 1 else 0)
        GLES30.glUniform1f(uOutScale, up.outScale)
        GLES30.glUniform1f(uOutZero, up.outZero.toFloat())
        bindQuad(quad)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        val t3 = System.nanoTime()

        sumRead += t1 - t0
        sumConv += conv
        sumNpu += npu
        sumUp += upl
        sumTotal += t3 - t0
        if (++frames == STATS_EVERY) publish(t3)
    }

    private fun publish(now: Long) {
        val n = frames.toDouble()
        fun ms(v: Long) = v / n / 1e6
        val fps = n / ((now - lastPublish) / 1e9)
        AiSettings.stats = String.format(
            Locale.US,
            "%.0f fps | per frame: read %.1f, convert %.1f, NPU %.1f, upload %.1f, total %.1f ms\n%d tiles: %dx%d -> %dx%d",
            fps, ms(sumRead), ms(sumConv), ms(sumNpu), ms(sumUp), ms(sumTotal),
            cols * rows, lowW, lowH, fullW, fullH,
        )
        frames = 0; sumRead = 0; sumConv = 0; sumNpu = 0; sumUp = 0; sumTotal = 0
        lastPublish = now
    }

    /**
     * Cuts tile (tx, ty) out of the read-back image into [tileFloats] (or [tileBytes] for uint8 models),
     * top row first, as the model expects.
     */
    private fun fillTile(tx: Int, ty: Int) {
        if (q) { fillTileU8(tx, ty); return }
        val w = up.inW
        val h = up.inH
        val plane = w * h
        val inv = 1f / 255f
        for (y in 0 until h) {
            val srcRow = (lowH - 1) - (ty * h + y) // GL rows run bottom to top
            var p = (srcRow * lowW + tx * w) * 4
            for (x in 0 until w) {
                val r = (rgba[p].toInt() and 0xFF) * inv
                val g = (rgba[p + 1].toInt() and 0xFF) * inv
                val b = (rgba[p + 2].toInt() and 0xFF) * inv
                if (up.nchwIn) {
                    val i = y * w + x
                    tileFloats[i] = r
                    tileFloats[plane + i] = g
                    tileFloats[2 * plane + i] = b
                } else {
                    val i = (y * w + x) * 3
                    tileFloats[i] = r
                    tileFloats[i + 1] = g
                    tileFloats[i + 2] = b
                }
                p += 4
            }
        }
    }

    private fun fillTileU8(tx: Int, ty: Int) {
        val w = up.inW
        val h = up.inH
        val plane = w * h
        val lut = up.inLut
        for (y in 0 until h) {
            val srcRow = (lowH - 1) - (ty * h + y) // GL rows run bottom to top
            var p = (srcRow * lowW + tx * w) * 4
            for (x in 0 until w) {
                val r = lut[rgba[p].toInt() and 0xFF]
                val g = lut[rgba[p + 1].toInt() and 0xFF]
                val b = lut[rgba[p + 2].toInt() and 0xFF]
                if (up.nchwIn) {
                    val i = y * w + x
                    tileBytes[i] = r
                    tileBytes[plane + i] = g
                    tileBytes[2 * plane + i] = b
                } else {
                    val i = (y * w + x) * 3
                    tileBytes[i] = r
                    tileBytes[i + 1] = g
                    tileBytes[i + 2] = b
                }
                p += 4
            }
        }
    }

    private fun uploadU8(out: ByteBuffer, tx: Int, ty: Int) {
        // Always copy into a native-order direct buffer we own: the runtime's buffer may be read-only.
        val need = out.remaining()
        var s = scratchU8
        if (s == null || s.capacity() < need) {
            s = ByteBuffer.allocateDirect(need).order(ByteOrder.nativeOrder())
            scratchU8 = s
        }
        s!!.clear()
        s.put(out.duplicate())
        s.rewind()
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        if (up.planarOut) {
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, tx * up.outW, ty * up.outH * 3, up.outW, up.outH * 3,
                GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, s,
            )
        } else {
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, tx * up.outW, ty * up.outH, up.outW, up.outH,
                GLES30.GL_RGB, GLES30.GL_UNSIGNED_BYTE, s,
            )
        }
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
    }

    private fun upload(out: FloatBuffer, tx: Int, ty: Int) {
        // Always copy into a native-order direct buffer we own: the runtime's buffer may be read-only.
        val need = out.remaining()
        var sc = scratch
        if (sc == null || sc.capacity() < need) {
            sc = ByteBuffer.allocateDirect(need * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            scratch = sc
        }
        sc!!.clear()
        sc.put(out.duplicate())
        sc.rewind()
        val fb = sc
        if (up.planarOut) {
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, tx * up.outW, ty * up.outH * 3, up.outW, up.outH * 3,
                GLES30.GL_RED, GLES30.GL_FLOAT, fb,
            )
        } else {
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, tx * up.outW, ty * up.outH, up.outW, up.outH,
                GLES30.GL_RGB, GLES30.GL_FLOAT, fb,
            )
        }
    }

    companion object {
        private const val STATS_EVERY = 30

        private const val COPY_VS = """#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform mat4 uStMatrix;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUv = (uStMatrix * vec4(aUv, 0.0, 1.0)).xy;
}
"""

        // Averages a 6x6 grid of bilinear taps per low-res pixel so the downscale does not alias.
        private const val COPY_FS = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES uTex;
uniform vec2 uFootprint;
in vec2 vUv;
out vec4 outColor;
void main() {
    vec3 acc = vec3(0.0);
    for (int j = 0; j < 6; j++) {
        for (int i = 0; i < 6; i++) {
            vec2 o = (vec2(float(i), float(j)) + 0.5) / 6.0 - 0.5;
            acc += texture(uTex, vUv + o * uFootprint).rgb;
        }
    }
    outColor = vec4(acc / 36.0, 1.0);
}
"""

        private const val PRESENT_VS = """#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUv = aUv;
}
"""

        private const val PRESENT_FS = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uAtlas;
uniform sampler2D uLow;
uniform ivec2 uTileOut;
uniform ivec2 uFull;
uniform int uPlanar;
uniform int uMode;
uniform int uQuant;
uniform float uOutScale;
uniform float uOutZero;
in vec2 vUv;
out vec4 outColor;

// uint8 atlas: the sampler returns q / 255, so recover q and apply (q - zeroPoint) * scale.
float deq(float t) {
    return uQuant == 1 ? (t * 255.0 - uOutZero) * uOutScale : t;
}

// p is a pixel of the full AI image, row 0 = top.
vec3 fetchAi(ivec2 p) {
    p = clamp(p, ivec2(0), uFull - ivec2(1));
    ivec2 t = p / uTileOut;
    ivec2 l = p - t * uTileOut;
    if (uPlanar == 1) {
        int x = t.x * uTileOut.x + l.x;
        int y = t.y * uTileOut.y * 3 + l.y;
        return vec3(deq(texelFetch(uAtlas, ivec2(x, y), 0).r),
                    deq(texelFetch(uAtlas, ivec2(x, y + uTileOut.y), 0).r),
                    deq(texelFetch(uAtlas, ivec2(x, y + 2 * uTileOut.y), 0).r));
    }
    vec3 q = texelFetch(uAtlas, p, 0).rgb;
    return vec3(deq(q.r), deq(q.g), deq(q.b));
}

void main() {
    vec3 col;
    bool useLow = (uMode == 2) || (uMode == 1 && vUv.x > 0.5);
    if (useLow) {
        col = texture(uLow, vUv).rgb;
    } else {
        vec2 pos = vec2(vUv.x, 1.0 - vUv.y) * vec2(uFull) - 0.5;
        vec2 f = fract(pos);
        ivec2 i = ivec2(floor(pos));
        vec3 a = fetchAi(i);
        vec3 b = fetchAi(i + ivec2(1, 0));
        vec3 c = fetchAi(i + ivec2(0, 1));
        vec3 d = fetchAi(i + ivec2(1, 1));
        col = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }
    if (uMode == 1 && abs(vUv.x - 0.5) < 0.0015) col = vec3(1.0, 0.2, 0.2);
    outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""
    }
}
