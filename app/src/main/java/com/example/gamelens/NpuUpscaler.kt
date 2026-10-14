package com.example.gamelens

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.system.Os
import java.nio.FloatBuffer

/**
 * Runs a fixed-size super-resolution ONNX model (QuickSRNetSmall: 128x128 -> 512x512, float I/O)
 * on the Qualcomm NPU through ONNX Runtime's QNN provider. Must be created and used on the GL thread.
 */
class NpuUpscaler(modelPath: String, nativeLibDir: String) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val opts: OrtSession.SessionOptions
    private val session: OrtSession
    private val inName: String
    private val inShape: LongArray

    val inW: Int
    val inH: Int
    val outW: Int
    val outH: Int
    /** True if the input tensor is [1,3,H,W]; false if [1,H,W,3]. */
    val nchwIn: Boolean
    /** True if the output tensor is [1,3,H,W] (planar); false if [1,H,W,3]. */
    val planarOut: Boolean
    val note: String

    init {
        // Lets the Hexagon DSP find its skel libraries, which the AAR extracts into the native lib dir.
        Os.setenv("ADSP_LIBRARY_PATH", "$nativeLibDir;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp", true)

        fun open(strict: Boolean): Pair<OrtSession, OrtSession.SessionOptions> {
            val o = OrtSession.SessionOptions()
            try {
                o.addQnn(mapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "burst"))
                if (strict) o.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                return env.createSession(modelPath, o) to o
            } catch (t: Throwable) {
                o.close()
                throw t
            }
        }

        var n = ""
        val pair = try {
            n = "NPU, no CPU fallback"
            open(true)
        } catch (t: Throwable) {
            n = "NPU with CPU fallback (strict failed: ${t.message?.take(100)})"
            open(false)
        }
        session = pair.first
        opts = pair.second
        note = n

        inName = session.inputNames.first()
        val ii = session.inputInfo[inName]!!.info as TensorInfo
        val oi = session.outputInfo[session.outputNames.first()]!!.info as TensorInfo
        require(ii.type == OnnxJavaType.FLOAT && oi.type == OnnxJavaType.FLOAT) {
            "Model must have float32 input and output (found ${ii.type} / ${oi.type})"
        }
        require(ii.shape.all { it > 0 } && oi.shape.all { it > 0 }) { "Model must have fixed shapes" }
        inShape = ii.shape
        val a = parse(ii.shape)
        val b = parse(oi.shape)
        nchwIn = a[0] == 1
        inW = a[1]; inH = a[2]
        planarOut = b[0] == 1
        outW = b[1]; outH = b[2]
    }

    /** Returns [isNchw (1/0), width, height] for a 4D shape that has 3 colour channels. */
    private fun parse(shape: LongArray): IntArray {
        require(shape.size == 4) { "Expected a 4D tensor, got ${shape.toList()}" }
        return when {
            shape[3] == 3L -> intArrayOf(0, shape[2].toInt(), shape[1].toInt())
            shape[1] == 3L -> intArrayOf(1, shape[3].toInt(), shape[2].toInt())
            else -> throw IllegalStateException("No 3-channel axis in ${shape.toList()}")
        }
    }

    fun describe() = "in ${inW}x$inH -> out ${outW}x$outH (${if (nchwIn) "NCHW" else "NHWC"} in, " +
        "${if (planarOut) "NCHW" else "NHWC"} out), $note"

    /** Runs one tile. [input] must hold the whole tile from position 0; [consume] must finish using the output before returning. */
    fun infer(input: FloatBuffer, consume: (FloatBuffer) -> Unit) {
        OnnxTensor.createTensor(env, input, inShape).use { t ->
            session.run(mapOf(inName to t)).use { res ->
                consume((res[0] as OnnxTensor).floatBuffer)
            }
        }
    }

    override fun close() {
        runCatching { session.close() }
        runCatching { opts.close() }
    }
}
