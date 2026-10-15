package com.example.gamelens

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.system.Os
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

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
    val nchwIn: Boolean
    val planarOut: Boolean
    /** Requested provider configuration; NPU execution is not verified by this class. */
    val note: String

    private val inputLock = Any()
    @Volatile private var closed = false
    private val scratch: FloatBuffer
    private val cachedInputTensor: OnnxTensor

    init {
        Os.setenv("ADSP_LIBRARY_PATH", "$nativeLibDir;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp", true)

        var sessionOpts: OrtSession.SessionOptions? = null
        var createdSession: OrtSession? = null
        var createdTensor: OnnxTensor? = null
        var configNote = ""
        var initFailure: Throwable? = null

        fun safeClose(resource: AutoCloseable?) {
            if (resource != null) {
                try {
                    resource.close()
                } catch (t: Throwable) {
                    if (initFailure == null) initFailure = t
                    else initFailure?.addSuppressed(t)
                }
            }
        }

        fun open(mode: Int): Pair<OrtSession, OrtSession.SessionOptions> {
            val o = OrtSession.SessionOptions()
            try {
                o.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                if (mode < 2) {
                    o.addQnn(
                        mapOf(
                            "backend_path" to "libQnnHtp.so",
                            "htp_performance_mode" to "burst",
                            "enable_htp_fp16_precision" to "1",
                        )
                    )
                    if (mode == 0) o.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                }
                return env.createSession(modelPath, o) to o
            } catch (t: Throwable) {
                o.close()
                throw t
            }
        }

        val errs = ArrayList<String>()
        var pair: Pair<OrtSession, OrtSession.SessionOptions>? = null
        for (mode in 0..2) {
            try {
                pair = open(mode)
                configNote = when (mode) {
                    0 -> "Requested QNN/HTP, no CPU fallback"
                    1 -> "Requested QNN/HTP with CPU fallback (strict failed: ${errs.joinToString()})"
                    else -> "Requested CPU only; NPU path failed: ${errs.joinToString()}"
                }
                break
            } catch (t: Throwable) {
                errs += "m$mode: ${t.message?.take(160)}"
            }
        }
        if (pair == null) throw IllegalStateException(errs.joinToString(" | "))

        createdSession = pair!!.first
        sessionOpts = pair!!.second

        try {
            session = createdSession!!
            opts = sessionOpts!!
            note = configNote

            inName = session.inputNames.first()
            val ii = session.inputInfo[inName]!!.info as TensorInfo
            val oi = session.outputInfo[session.outputNames.first()]!!.info as TensorInfo

            require(ii.type == OnnxJavaType.FLOAT && oi.type == OnnxJavaType.FLOAT) {
                "Model must have float32 input and output (found ${ii.type} / ${oi.type})"
            }
            require(ii.shape.size == 4 && oi.shape.size == 4) {
                "Model input/output must be rank 4 (in ${ii.shape.toList()}, out ${oi.shape.toList()})"
            }
            require(ii.shape[0] == 1L && oi.shape[0] == 1L) {
                "Model batch dimension must be 1 (in ${ii.shape.toList()}, out ${oi.shape.toList()})"
            }

            val i1 = positiveInt(ii.shape[1], "input dim 1")
            val i2 = positiveInt(ii.shape[2], "input dim 2")
            val i3 = positiveInt(ii.shape[3], "input dim 3")
            val inElements = Math.multiplyExact(Math.multiplyExact(i1, i2), i3)
            val inByteSize = Math.multiplyExact(inElements, 4)

            val o1 = positiveInt(oi.shape[1], "output dim 1")
            val o2 = positiveInt(oi.shape[2], "output dim 2")
            val o3 = positiveInt(oi.shape[3], "output dim 3")
            val outElements = Math.multiplyExact(Math.multiplyExact(o1, o2), o3)
            require(outElements > 0) { "Output spatial/channel dimensions must be positive" }

            inShape = ii.shape

            val a = parse(ii.shape)
            val b = parse(oi.shape)
            nchwIn = a[0] == 1
            inW = a[1]
            inH = a[2]
            planarOut = b[0] == 1
            outW = b[1]
            outH = b[2]

            val bb = ByteBuffer.allocateDirect(inByteSize).order(ByteOrder.nativeOrder())
            scratch = bb.asFloatBuffer()
            require(scratch.capacity() == inElements) { "Scratch buffer capacity mismatch" }

            createdTensor = OnnxTensor.createTensor(env, scratch, inShape)
            cachedInputTensor = createdTensor!!
        } catch (t: Throwable) {
            safeClose(createdTensor)
            safeClose(createdSession)
            safeClose(sessionOpts)
            initFailure?.let { t.addSuppressed(it) }
            throw t
        }
    }

    fun describe() = "in ${inW}x$inH -> out ${outW}x$outH (${if (nchwIn) "NCHW" else "NHWC"} in, " +
        "${if (planarOut) "NCHW" else "NHWC"} out), $note"

    fun infer(input: FloatBuffer, consume: (FloatBuffer) -> Unit): Unit = synchronized(inputLock) {
        check(!closed) { IllegalStateException("NpuUpscaler is closed") }

        val dup = input.duplicate()
        val rem = dup.remaining()
        require(rem == scratch.capacity()) {
            "Tile size mismatch: expected ${scratch.capacity()} floats, got $rem"
        }

        scratch.clear()
        scratch.put(dup)
        scratch.rewind()
        // Caller position/limit preserved via duplicate().

        session.run(mapOf(inName to cachedInputTensor)).use { results ->
            val src = (results[0] as OnnxTensor).floatBuffer
            val copy = FloatBuffer.allocate(src.remaining()).also { it.put(src); it.rewind() }
            consume(copy)
        }
    }

    override fun close(): Unit = synchronized(inputLock) {
        if (closed) return@synchronized
        closed = true
        runCatching { cachedInputTensor.close() }
        runCatching { session.close() }
        runCatching { opts.close() }
    }

    private fun positiveInt(v: Long, label: String): Int {
        require(v > 0 && v <= Int.MAX_VALUE) { "$label must be positive and fit in Int, got $v" }
        return v.toInt()
    }

    private fun parse(shape: LongArray): IntArray {
        require(shape.size == 4) { "Expected a 4D tensor, got ${shape.toList()}" }
        return when {
            shape[3] == 3L && shape[1] != 3L -> intArrayOf(0, shape[2].toInt(), shape[1].toInt())
            shape[1] == 3L && shape[3] != 3L -> intArrayOf(1, shape[3].toInt(), shape[2].toInt())
            else -> throw IllegalStateException("Ambiguous or missing 3-channel axis in ${shape.toList()}")
        }
    }
}
