package com.example.gamelens

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

/**
 * Answers one question before we touch the live pipeline: is a neural upscaler fast enough on THIS phone?
 *
 * It loads Qualcomm's QuickSRNetSmall, times it on CPU and GPU, scales the result up to a full-screen
 * frame-time estimate, and compares its output with plain bilinear upscaling on a generated test image.
 */
class BenchmarkActivity : ComponentActivity() {

    private lateinit var logView: TextView
    private lateinit var runButton: Button
    private lateinit var images: LinearLayout
    private val logText = StringBuilder()

    private class TensorLayout(val nhwc: Boolean, val h: Int, val w: Int) {
        val count get() = h * w * 3
    }

    private class Timing(val medianMs: Double, val meanMs: Double)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 2, pad, pad)
        }
        root.addView(TextView(this).apply {
            text = "AI upscaler benchmark"
            textSize = 22f
        })
        root.addView(TextView(this).apply {
            text = "Tests QuickSRNetSmall (Qualcomm, BSD-3) on this phone: speed on CPU and GPU, and " +
                "quality against plain bilinear upscaling on a generated test image."
        })
        runButton = Button(this).apply {
            text = "Run benchmark"
            setOnClickListener { startRun() }
        }
        root.addView(runButton)
        images = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        root.addView(images)
        logView = TextView(this).apply {
            setTextIsSelectable(true)
            typeface = Typeface.MONOSPACE
            textSize = 12f
        }
        root.addView(logView)
        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun startRun() {
        runButton.isEnabled = false
        logText.setLength(0)
        logView.text = ""
        images.removeAllViews()
        Thread {
            try {
                benchmark()
            } catch (t: Throwable) {
                say("ERROR: ${t.javaClass.simpleName}: ${t.message}")
            }
            runOnUiThread { runButton.isEnabled = true }
        }.start()
    }

    private fun say(line: String) {
        runOnUiThread {
            logText.append(line).append('\n')
            logView.text = logText.toString()
        }
    }

    // ---------------------------------------------------------------- benchmark

    private fun benchmark() {
        say("Device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.SOC_MODEL})")
        say("Android ${Build.VERSION.RELEASE}")
        if (!runCatching { assets.open(MODEL_ASSET).close() }.isSuccess) {
            say("Model file missing from the APK. The GitHub build step that downloads it did not run.")
            return
        }

        val cpu = Interpreter(loadModel(), Interpreter.Options().setNumThreads(4))
        try {
            val inT = cpu.getInputTensor(0)
            val outT = cpu.getOutputTensor(0)
            check(inT.dataType() == DataType.FLOAT32 && outT.dataType() == DataType.FLOAT32) {
                "Only float32 models are supported here (got ${inT.dataType()} / ${outT.dataType()})"
            }
            val inLay = layoutOf(inT.shape())
            val outLay = layoutOf(outT.shape())
            say("Input tensor ${inT.shape().toList()}, output ${outT.shape().toList()}")
            say("Model scale: x%.2f".format(outLay.w.toDouble() / inLay.w))
            say("")

            val cpuTiming = time(cpu, newBuf(inLay.count), newBuf(outLay.count))
            say("CPU (4 threads): median %.2f ms, mean %.2f ms per %dx%d tile"
                .format(cpuTiming.medianMs, cpuTiming.meanMs, inLay.w, inLay.h))

            val gpuTiming = runGpu(inLay, outLay)

            say("")
            quality(cpu, inLay, outLay)
            estimate(inLay, outLay, cpuTiming, gpuTiming)
        } finally {
            cpu.close()
        }
    }

    private fun runGpu(inLay: TensorLayout, outLay: TensorLayout): Timing? {
        var delegate: GpuDelegate? = null
        var interp: Interpreter? = null
        return try {
            delegate = GpuDelegate()
            interp = Interpreter(loadModel(), Interpreter.Options().addDelegate(delegate))
            val timing = time(interp, newBuf(inLay.count), newBuf(outLay.count))
            say("GPU: median %.2f ms, mean %.2f ms per %dx%d tile"
                .format(timing.medianMs, timing.meanMs, inLay.w, inLay.h))
            timing
        } catch (e: Throwable) {
            say("GPU delegate unavailable: ${e.javaClass.simpleName}: ${e.message}")
            null
        } finally {
            runCatching { interp?.close() }
            runCatching { delegate?.close() }
        }
    }

    private fun time(
        interp: Interpreter,
        inBuf: ByteBuffer,
        outBuf: ByteBuffer,
        warmup: Int = 5,
        iterations: Int = 40,
    ): Timing {
        repeat(warmup) {
            inBuf.rewind(); outBuf.rewind()
            interp.run(inBuf, outBuf)
        }
        val ms = DoubleArray(iterations)
        for (i in 0 until iterations) {
            inBuf.rewind(); outBuf.rewind()
            val start = System.nanoTime()
            interp.run(inBuf, outBuf)
            ms[i] = (System.nanoTime() - start) / 1e6
        }
        ms.sort()
        return Timing(ms[iterations / 2], ms.average())
    }

    private fun estimate(inLay: TensorLayout, outLay: TensorLayout, cpu: Timing, gpu: Timing?) {
        val bounds = windowManager.maximumWindowMetrics.bounds
        val long = max(bounds.width(), bounds.height())
        val short = min(bounds.width(), bounds.height())
        val scale = outLay.w.toDouble() / inLay.w
        val inLong = long / scale
        val inShort = short / scale
        val tiles = ceil(inLong / inLay.w) * ceil(inShort / inLay.h)

        say("Rough full-screen estimate")
        say("  Screen ${long}x${short}; the game would render %dx%d (screen / %.1f)"
            .format(inLong.toInt(), inShort.toInt(), scale))
        say("  That is about %.0f tiles of %dx%d".format(tiles, inLay.w, inLay.h))
        fun line(name: String, t: Timing) {
            val frameMs = tiles * t.medianMs
            say("  $name: ~%.1f ms per frame -> %s".format(frameMs, verdict(frameMs)))
        }
        line("CPU", cpu)
        if (gpu != null) line("GPU", gpu)
        say("  (Pessimistic: runs tile by tile. Real use would feed bigger pieces, but it also needs")
        say("   screen capture and the game itself sharing the same chip.)")
    }

    private fun verdict(frameMs: Double) = when {
        frameMs <= 8.3 -> "fits a 120 fps budget"
        frameMs <= 16.7 -> "fits a 60 fps budget"
        frameMs <= 33.3 -> "30 fps at best"
        else -> "too slow for live use"
    }

    // ---------------------------------------------------------------- quality check

    private fun quality(interp: Interpreter, inLay: TensorLayout, outLay: TensorLayout) {
        val ref = makeReference(outLay.w, outLay.h)
        val low = Bitmap.createScaledBitmap(ref, inLay.w, inLay.h, true)
        val bilinear = Bitmap.createScaledBitmap(low, outLay.w, outLay.h, true)

        val inBuf = newBuf(inLay.count)
        val outBuf = newBuf(outLay.count)
        fillInput(inBuf, low, inLay)
        inBuf.rewind(); outBuf.rewind()
        interp.run(inBuf, outBuf)
        val ai = toBitmap(outBuf, outLay)

        val pBilinear = psnr(ref, bilinear)
        val pAi = psnr(ref, ai)
        say("Test image (higher dB = closer to the original):")
        say("  Bilinear upscale: %.2f dB".format(pBilinear))
        say("  QuickSRNetSmall:  %.2f dB".format(pAi))
        say("  (Rough sanity check on a synthetic image, not a verdict on real game frames.)")

        runOnUiThread {
            addImage("Original", ref)
            addImage("Bilinear %.1f dB".format(pBilinear), bilinear)
            addImage("AI %.1f dB".format(pAi), ai)
        }
    }

    private fun addImage(label: String, bmp: Bitmap) {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        col.addView(ImageView(this).apply {
            setImageBitmap(bmp)
            adjustViewBounds = true
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            )
        })
        col.addView(TextView(this).apply {
            text = label
            textSize = 11f
        })
        images.addView(col)
    }

    private fun makeReference(w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        val p = Paint(Paint.ANTI_ALIAS_FLAG)
        val wf = w.toFloat()
        val hf = h.toFloat()

        p.shader = LinearGradient(0f, 0f, wf, hf, 0xFF203A5C.toInt(), 0xFFE0A050.toInt(), Shader.TileMode.CLAMP)
        c.drawRect(0f, 0f, wf, hf, p)
        p.shader = null

        // thin grid
        p.style = Paint.Style.STROKE
        p.color = 0xFFFFFFFF.toInt()
        p.strokeWidth = max(1f, wf / 250f)
        for (i in 1 until 8) {
            c.drawLine(wf * i / 8f, 0f, wf * i / 8f, hf, p)
            c.drawLine(0f, hf * i / 8f, wf, hf * i / 8f, p)
        }
        // circles and diagonals
        p.color = 0xFF101010.toInt()
        for (r in 1..4) c.drawCircle(wf * 0.7f, hf * 0.3f, wf * 0.05f * r, p)
        p.color = 0xFFFFE070.toInt()
        for (i in 0 until 6) c.drawLine(0f, hf * (0.5f + i * 0.03f), wf * 0.5f, hf * (0.95f - i * 0.03f), p)
        // checker patch
        p.style = Paint.Style.FILL
        val cell = max(2f, wf / 40f)
        for (yy in 0 until 4) for (xx in 0 until 4) {
            p.color = if ((xx + yy) % 2 == 0) 0xFF000000.toInt() else 0xFFFFFFFF.toInt()
            val x0 = wf * 0.62f + xx * cell
            val y0 = hf * 0.62f + yy * cell
            c.drawRect(x0, y0, x0 + cell, y0 + cell, p)
        }
        // text
        p.color = 0xFFFFFFFF.toInt()
        p.typeface = Typeface.DEFAULT_BOLD
        p.textSize = wf / 9f
        c.drawText("GameLens", wf * 0.06f, hf * 0.45f, p)
        p.typeface = Typeface.DEFAULT
        p.textSize = wf / 20f
        c.drawText("AI upscale test 0123456789", wf * 0.06f, hf * 0.56f, p)
        return bmp
    }

    // ---------------------------------------------------------------- tensor helpers

    private fun layoutOf(shape: IntArray): TensorLayout {
        require(shape.size == 4) { "Expected a 4-D tensor, got ${shape.toList()}" }
        val lay = when {
            shape[3] == 3 -> TensorLayout(true, shape[1], shape[2])
            shape[1] == 3 -> TensorLayout(false, shape[2], shape[3])
            else -> error("Unsupported tensor shape ${shape.toList()}")
        }
        check(lay.h >= 8 && lay.w >= 8) {
            "Model has a dynamic input size (${shape.toList()}), which this benchmark does not handle"
        }
        return lay
    }

    private fun loadModel(): ByteBuffer {
        val bytes = assets.open(MODEL_ASSET).use { it.readBytes() }
        val buf = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buf.put(bytes)
        buf.rewind()
        return buf
    }

    private fun newBuf(floats: Int): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(floats * 4).order(ByteOrder.nativeOrder())
        val f = buf.asFloatBuffer()
        for (i in 0 until floats) f.put(i, 0.5f)
        return buf
    }

    private fun fillInput(buf: ByteBuffer, bmp: Bitmap, lay: TensorLayout) {
        val px = IntArray(lay.w * lay.h)
        bmp.getPixels(px, 0, lay.w, 0, 0, lay.w, lay.h)
        buf.rewind() // the float view starts at the buffer's current position
        val f = buf.asFloatBuffer()
        val plane = lay.w * lay.h
        for (y in 0 until lay.h) for (x in 0 until lay.w) {
            val p = px[y * lay.w + x]
            val r = ((p shr 16) and 255) / 255f
            val g = ((p shr 8) and 255) / 255f
            val b = (p and 255) / 255f
            if (lay.nhwc) {
                val i = (y * lay.w + x) * 3
                f.put(i, r); f.put(i + 1, g); f.put(i + 2, b)
            } else {
                val i = y * lay.w + x
                f.put(i, r); f.put(plane + i, g); f.put(2 * plane + i, b)
            }
        }
    }

    private fun toBitmap(buf: ByteBuffer, lay: TensorLayout): Bitmap {
        buf.rewind() // the float view starts at the buffer's current position
        val f = buf.asFloatBuffer()
        val plane = lay.w * lay.h
        val px = IntArray(plane)
        for (y in 0 until lay.h) for (x in 0 until lay.w) {
            val r: Float
            val g: Float
            val b: Float
            if (lay.nhwc) {
                val i = (y * lay.w + x) * 3
                r = f.get(i); g = f.get(i + 1); b = f.get(i + 2)
            } else {
                val i = y * lay.w + x
                r = f.get(i); g = f.get(plane + i); b = f.get(2 * plane + i)
            }
            px[y * lay.w + x] = (0xFF shl 24) or (to8(r) shl 16) or (to8(g) shl 8) or to8(b)
        }
        return Bitmap.createBitmap(px, lay.w, lay.h, Bitmap.Config.ARGB_8888)
    }

    private fun to8(v: Float): Int = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()

    private fun psnr(a: Bitmap, b: Bitmap): Double {
        val n = a.width * a.height
        val pa = IntArray(n)
        val pb = IntArray(n)
        a.getPixels(pa, 0, a.width, 0, 0, a.width, a.height)
        b.getPixels(pb, 0, b.width, 0, 0, b.width, b.height)
        var se = 0.0
        for (i in 0 until n) {
            for (shift in intArrayOf(16, 8, 0)) {
                val d = ((pa[i] shr shift) and 255) - ((pb[i] shr shift) and 255)
                se += (d * d).toDouble()
            }
        }
        val mse = se / (n * 3.0)
        return if (mse <= 0.0) 99.0 else 10.0 * log10(255.0 * 255.0 / mse)
    }

    companion object {
        const val MODEL_ASSET = "quicksrnet_small_float.tflite"
    }
}
