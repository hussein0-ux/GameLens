package com.example.gamelens

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * CPU reference for the Enhance math in GlPipeline (GSR_COPY_FRAGMENT_SRC edge AA) and
 * sgsr1_mobile.frag (clarity, vibrance). The shaders are line-by-line translations of these functions.
 */
private class Img(val w: Int, val h: Int, val rgb: FloatArray = FloatArray(w * h * 3)) {
    fun px(x: Int, y: Int, r: Float, g: Float, b: Float) {
        val i = (y * w + x) * 3; rgb[i] = r; rgb[i + 1] = g; rgb[i + 2] = b
    }
    fun gray(x: Int, y: Int, v: Float) = px(x, y, v, v, v)
    private fun at(x: Int, y: Int, c: Int) = rgb[(y.coerceIn(0, h - 1) * w + x.coerceIn(0, w - 1)) * 3 + c]
    /** Bilinear sample at texel-centre coordinates, like the GPU's GL_LINEAR with clamp to edge. */
    fun sample(px: Float, py: Float): FloatArray {
        val fx = px - 0.5f; val fy = py - 0.5f
        val x0 = Math.floor(fx.toDouble()).toInt(); val y0 = Math.floor(fy.toDouble()).toInt()
        val tx = fx - x0; val ty = fy - y0
        return FloatArray(3) { c ->
            val top = at(x0, y0, c) * (1 - tx) + at(x0 + 1, y0, c) * tx
            val bot = at(x0, y0 + 1, c) * (1 - tx) + at(x0 + 1, y0 + 1, c) * tx
            top * (1 - ty) + bot * ty
        }
    }
}

private fun luma(c: FloatArray) = 0.299f * c[0] + 0.587f * c[1] + 0.114f * c[2]

/** FXAA 3 "console" directional blur, skipped on low-contrast pixels, mixed in by [strength]. */
private fun edgeAa(img: Img, x: Int, y: Int, strength: Float): FloatArray {
    val cx = x + 0.5f; val cy = y + 0.5f
    val m = img.sample(cx, cy)
    if (strength <= 0f) return m
    val nw = luma(img.sample(cx - 1, cy - 1)); val ne = luma(img.sample(cx + 1, cy - 1))
    val sw = luma(img.sample(cx - 1, cy + 1)); val se = luma(img.sample(cx + 1, cy + 1))
    val lm = luma(m)
    val lumaMin = min(lm, min(min(nw, ne), min(sw, se)))
    val lumaMax = max(lm, max(max(nw, ne), max(sw, se)))
    if (lumaMax - lumaMin < max(0.0312f, lumaMax * 0.125f)) return m
    var dx = -((nw + ne) - (sw + se))
    var dy = (nw + sw) - (ne + se)
    val dirReduce = max((nw + ne + sw + se) * 0.25f * 0.125f, 1f / 128f)
    val rcp = 1f / (min(abs(dx), abs(dy)) + dirReduce)
    dx = (dx * rcp).coerceIn(-8f, 8f); dy = (dy * rcp).coerceIn(-8f, 8f)
    val a1 = img.sample(cx + dx * (1f / 3f - 0.5f), cy + dy * (1f / 3f - 0.5f))
    val a2 = img.sample(cx + dx * (2f / 3f - 0.5f), cy + dy * (2f / 3f - 0.5f))
    val b1 = img.sample(cx - dx * 0.5f, cy - dy * 0.5f)
    val b2 = img.sample(cx + dx * 0.5f, cy + dy * 0.5f)
    val rgbA = FloatArray(3) { 0.5f * (a1[it] + a2[it]) }
    val rgbB = FloatArray(3) { rgbA[it] * 0.5f + 0.25f * (b1[it] + b2[it]) }
    val lb = luma(rgbB)
    val aa = if (lb < lumaMin || lb > lumaMax) rgbA else rgbB
    return FloatArray(3) { m[it] + (aa[it] - m[it]) * strength }
}

/** Pushes muted colours further than vivid ones; greys stay grey. */
private fun vibrance(c: FloatArray, amount: Float): FloatArray {
    val sat = max(c[0], max(c[1], c[2])) - min(c[0], min(c[1], c[2]))
    val k = 1f + amount * (1f - sat)
    val l = luma(c)
    return FloatArray(3) { (l + (c[it] - l) * k).coerceIn(0f, 1f) }
}

/** Local contrast: adds the difference between the pixel and a wide blur of its surroundings. */
private fun clarity(c: FloatArray, blurLuma: Float, amount: Float): FloatArray {
    val d = (luma(c) - blurLuma) * amount
    return FloatArray(3) { (c[it] + d).coerceIn(0f, 1f) }
}

private fun sat(c: FloatArray) = max(c[0], max(c[1], c[2])) - min(c[0], min(c[1], c[2]))

class EnhanceMathTest {
    private fun staircase(): Img {
        // dark left, bright right, diagonal staircase edge with 1 px steps
        val img = Img(24, 24)
        for (y in 0 until 24) for (x in 0 until 24) img.gray(x, y, if (x > y) 1f else 0f)
        return img
    }

    @Test fun aaStrengthZeroIsIdentity() {
        val img = staircase()
        for (y in 0 until 24) for (x in 0 until 24) {
            val out = edgeAa(img, x, y, 0f); val src = img.sample(x + 0.5f, y + 0.5f)
            for (c in 0..2) assertEquals(src[c], out[c], 1e-6f)
        }
    }

    @Test fun aaLeavesFlatAreasAlone() {
        val img = Img(16, 16)
        for (y in 0 until 16) for (x in 0 until 16) img.px(x, y, 0.4f, 0.5f, 0.6f)
        val out = edgeAa(img, 8, 8, 1f)
        assertEquals(0.4f, out[0], 1e-6f); assertEquals(0.6f, out[2], 1e-6f)
    }

    @Test fun aaLeavesLowContrastTextureAlone() {
        val img = Img(16, 16)
        for (y in 0 until 16) for (x in 0 until 16) img.gray(x, y, 0.50f + 0.01f * ((x + y) % 2))
        val out = edgeAa(img, 8, 8, 1f)
        assertEquals(img.sample(8.5f, 8.5f)[0], out[0], 1e-6f)
    }

    @Test fun aaSoftensStaircaseCornersAndNothingFarAway() {
        val img = staircase()
        var softened = 0
        for (y in 4 until 20) {
            val x = y + 1 // first bright pixel on the staircase, value 1.0 in the source
            val v = edgeAa(img, x, y, 1f)[0]
            if (v < 0.95f && v > 0.0f) softened++
        }
        assertEquals(true, softened >= 12)
        // 5+ px away from the edge the image is flat, so nothing changes
        assertEquals(1f, edgeAa(img, 20, 3, 1f)[0], 1e-6f)
        assertEquals(0f, edgeAa(img, 3, 20, 1f)[0], 1e-6f)
    }

    @Test fun aaOutputStaysInsideImageRange() {
        val img = staircase()
        for (y in 0 until 24) for (x in 0 until 24) {
            val v = edgeAa(img, x, y, 1f)[0]
            assertEquals(true, v >= -1e-5f && v <= 1f + 1e-5f)
        }
    }

    @Test fun vibranceZeroIsIdentityAndGreyStaysGrey() {
        val c = floatArrayOf(0.5f, 0.4f, 0.3f)
        val same = vibrance(c, 0f)
        for (i in 0..2) assertEquals(c[i], same[i], 1e-6f)
        val g = vibrance(floatArrayOf(0.5f, 0.5f, 0.5f), 1f)
        for (i in 0..2) assertEquals(0.5f, g[i], 1e-6f)
    }

    @Test fun vibranceBoostsMutedMoreThanVivid() {
        val muted = floatArrayOf(0.55f, 0.50f, 0.45f)
        val vivid = floatArrayOf(0.90f, 0.30f, 0.10f)
        val gainMuted = sat(vibrance(muted, 0.5f)) / sat(muted)
        val gainVivid = sat(vibrance(vivid, 0.5f)) / sat(vivid)
        assertEquals(true, gainMuted > gainVivid)
        assertEquals(true, gainMuted > 1.3f)
    }

    @Test fun clarityZeroIsIdentityAndFlatStaysFlat() {
        val c = floatArrayOf(0.5f, 0.4f, 0.3f)
        val same = clarity(c, 0.1f, 0f)
        for (i in 0..2) assertEquals(c[i], same[i], 1e-6f)
        val flat = clarity(c, luma(c), 1f)
        for (i in 0..2) assertEquals(c[i], flat[i], 1e-6f)
    }

    @Test fun clarityPushesAwayFromSurroundings() {
        val bright = clarity(floatArrayOf(0.6f, 0.6f, 0.6f), 0.5f, 1f)
        val dark = clarity(floatArrayOf(0.4f, 0.4f, 0.4f), 0.5f, 1f)
        assertEquals(true, bright[0] > 0.6f)
        assertEquals(true, dark[0] < 0.4f)
    }
}
