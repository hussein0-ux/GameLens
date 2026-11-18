package com.example.gamelens

import android.opengl.GLES11Ext
import android.opengl.GLES30
import java.nio.ByteBuffer

/**
 * GL side of the neural mode (GL thread only):
 *  1. crop the captured frame to the model size,
 *  2. pack it on the GPU into the model's planar int8 input,
 *  3. read it back through two pixel buffers, so the GL thread never waits for the GPU,
 *  4. hand it to [NeuralStage], and upload the newest NPU result into [planarTex] for the final draw.
 */
class NeuralPass(
    private val stage: NeuralStage,
    val info: NeuralModelInfo,
    private val captureW: Int,
    private val captureH: Int,
    private val build: (String, String) -> Int,
    private val drawQuad: () -> Unit,
) {
    val outW = info.w * info.scale
    val outH = info.h * info.scale
    /** True once at least one NPU result has been uploaded. */
    var hasResult = false
        private set
    val planarTex = IntArray(1)

    private val packW = info.w / 4
    private val packH = info.h * 3
    private val packBytes = packW * packH * 4

    private val cropFbo = IntArray(1)
    private val cropTex = IntArray(1)
    private val packFbo = IntArray(1)
    private val packTex = IntArray(1)
    private val pbo = IntArray(2)
    private var cropProg = 0
    private var packProg = 0
    private var frame = 0

    fun init() {
        makeTarget(cropFbo, cropTex, info.w, info.h)
        makeTarget(packFbo, packTex, packW, packH)

        GLES30.glGenBuffers(2, pbo, 0)
        for (b in pbo) {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, b)
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, packBytes, null, GLES30.GL_STREAM_READ)
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)

        // Planar output, 3 planes stacked vertically. R8_SNORM keeps the raw signed bytes and is filterable.
        GLES30.glGenTextures(1, planarTex, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, planarTex[0])
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8_SNORM, outW, outH * 3, 0, GLES30.GL_RED, GLES30.GL_BYTE, null)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        cropProg = build(CROP_VERTEX, CROP_FRAGMENT)
        packProg = build(PACK_VERTEX, PACK_FRAGMENT)
        check(GLES30.glGetError() == GLES30.GL_NO_ERROR) { "GL error while setting up the neural pass" }
    }

    private fun makeTarget(fbo: IntArray, tex: IntArray, w: Int, h: Int) {
        GLES30.glGenTextures(1, tex, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0])
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glGenFramebuffers(1, fbo, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo[0])
        GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, tex[0], 0)
        check(GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER) == GLES30.GL_FRAMEBUFFER_COMPLETE) { "Framebuffer incomplete ${w}x$h" }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
    }

    /** Takes one captured frame: crop, pack, and pass the previous frame's packed bytes to the NPU. */
    fun process(oesTex: Int, stMatrix: FloatArray) {
        // 1) crop the capture to the model size (exact texel centres, so no resampling blur)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, cropFbo[0])
        GLES30.glViewport(0, 0, info.w, info.h)
        GLES30.glUseProgram(cropProg)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(cropProg, "uStMatrix"), 1, false, stMatrix, 0)
        GLES30.glUniform4f(
            GLES30.glGetUniformLocation(cropProg, "uCrop"),
            info.w.toFloat() / captureW, info.h.toFloat() / captureH,
            ((captureW - info.w) / 2).toFloat() / captureW, ((captureH - info.h) / 2).toFloat() / captureH,
        )
        drawQuad()

        // 2) pack into the model's planar int8 input
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, packFbo[0])
        GLES30.glViewport(0, 0, packW, packH)
        GLES30.glUseProgram(packProg)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, cropTex[0])
        GLES30.glUniform1i(GLES30.glGetUniformLocation(packProg, "src"), 0)
        GLES30.glUniform2f(GLES30.glGetUniformLocation(packProg, "uQ"), info.inScale, info.inZp.toFloat())
        GLES30.glUniform1i(GLES30.glGetUniformLocation(packProg, "uTH"), info.h)
        drawQuad()

        // 3) start an asynchronous read of this frame, and collect the previous frame's bytes
        val cur = frame and 1
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbo[cur])
        GLES30.glReadPixels(0, 0, packW, packH, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)
        if (frame > 0) {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbo[1 - cur])
            val mapped = GLES30.glMapBufferRange(GLES30.GL_PIXEL_PACK_BUFFER, 0, packBytes, GLES30.GL_MAP_READ_BIT)
            if (mapped is ByteBuffer) {
                stage.submit(mapped)
                GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)
            }
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        frame++
    }

    /** Uploads the newest NPU result, if there is one. */
    fun uploadLatest() {
        val res = stage.takeLatest() ?: return
        check(res.size == outW * outH * 3) { "Unexpected NPU output size ${res.size}, expected ${outW * outH * 3}" }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, planarTex[0])
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glTexSubImage2D(
            GLES30.GL_TEXTURE_2D, 0, 0, 0, outW, outH * 3, GLES30.GL_RED, GLES30.GL_BYTE, ByteBuffer.wrap(res),
        )
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        hasResult = true
    }

    fun release() {
        GLES30.glDeleteFramebuffers(1, cropFbo, 0)
        GLES30.glDeleteFramebuffers(1, packFbo, 0)
        GLES30.glDeleteTextures(1, cropTex, 0)
        GLES30.glDeleteTextures(1, packTex, 0)
        GLES30.glDeleteTextures(1, planarTex, 0)
        GLES30.glDeleteBuffers(2, pbo, 0)
        if (cropProg != 0) GLES30.glDeleteProgram(cropProg)
        if (packProg != 0) GLES30.glDeleteProgram(packProg)
    }

    private companion object {
        // Crop: draw the part of the capture that the model sees (uCrop = scale xy, offset xy in 0..1 picture units).
        const val CROP_VERTEX = """#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform mat4 uStMatrix;
uniform vec4 uCrop;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUv = (uStMatrix * vec4(aUv * uCrop.xy + uCrop.zw, 0.0, 1.0)).xy;
}
"""
        const val CROP_FRAGMENT = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES uTex;
in vec2 vUv;
out vec4 outColor;
void main() { outColor = vec4(texture(uTex, vUv).rgb, 1.0); }
"""

        const val PACK_VERTEX = """#version 300 es
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
"""
        // Output texel (x, y) holds the model-input bytes x*4..x*4+3 of row (y % H) of channel (y / H).
        // Row 0 of the model input is the TOP of the picture, which is the last row of the cropped texture.
        // int8: values are quantised with the +128 shifted zero point, then the sign bit is flipped back.
        const val PACK_FRAGMENT = """#version 300 es
precision highp float;
uniform highp sampler2D src;
uniform vec2 uQ;
uniform int uTH;
out vec4 outColor;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int ch = p.y / uTH;
    int row = p.y - ch * uTH;
    vec4 o = vec4(0.0);
    for (int k = 0; k < 4; k++) {
        vec3 c = texelFetch(src, ivec2(p.x * 4 + k, uTH - 1 - row), 0).rgb;
        float v = ch == 0 ? c.r : (ch == 1 ? c.g : c.b);
        float q = floor(clamp(v / uQ.x + uQ.y, 0.0, 255.0) + 0.5);
        q = q >= 128.0 ? q - 128.0 : q + 128.0;
        o[k] = q / 255.0;
    }
    outColor = o;
}
"""
    }
}
