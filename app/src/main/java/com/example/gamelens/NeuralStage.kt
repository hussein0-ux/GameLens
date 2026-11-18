package com.example.gamelens

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/** What the GL side needs to know about the model. Zero points already carry the +128 shift used for int8. */
class NeuralModelInfo(
    val w: Int, val h: Int, val scale: Int,
    val inScale: Float, val inZp: Int, val outScale: Float, val outZp: Int,
)

/** Reads shapes and quantisation from an int8 planar (NCHW) AI Hub model. Throws with a clear message otherwise. */
fun readNeuralModelInfo(file: File): NeuralModelInfo {
    val bytes = file.readBytes()
    require(bytes.size > 8 && bytes[4] == 'T'.code.toByte() && bytes[5] == 'F'.code.toByte() &&
        bytes[6] == 'L'.code.toByte() && bytes[7] == '3'.code.toByte()) { "${file.name} is not a .tflite model" }
    val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
    bb.put(bytes)
    bb.rewind()
    val itp = Interpreter(bb, Interpreter.Options().apply { setNumThreads(1) })
    try {
        val ti = itp.getInputTensor(0)
        val to = itp.getOutputTensor(0)
        val i = ti.shape()
        val o = to.shape()
        require(i.size == 4 && o.size == 4 && i[1] == 3 && o[1] == 3) {
            "Neural mode needs a planar NCHW model, got ${i.toList()} -> ${o.toList()}"
        }
        require(ti.dataType() == DataType.INT8 && to.dataType() == DataType.INT8) {
            "Neural mode needs an int8 (w8a8) model, got ${ti.dataType()} -> ${to.dataType()}"
        }
        val h = i[2]
        val w = i[3]
        require(w % 4 == 0) { "Model width $w must be divisible by 4" }
        require(o[2] % h == 0 && o[3] % w == 0 && o[2] / h == o[3] / w) { "Unsupported output shape ${o.toList()}" }
        return NeuralModelInfo(
            w, h, o[2] / h,
            ti.quantizationParams().scale, ti.quantizationParams().zeroPoint + 128,
            to.quantizationParams().scale, to.quantizationParams().zeroPoint + 128,
        )
    } finally {
        itp.close()
    }
}

/**
 * Runs the model on the Hexagon NPU on two worker threads so the 120 fps budget is about the NPU time alone:
 *  - "npu-run" writes the input and runs the graph into one of two output buffer sets;
 *  - "npu-read" copies the finished output set to a byte array while the next run is already going.
 * Frames are never queued: [submit] refuses a frame while the NPU is busy, and [takeLatest] only hands out the newest result.
 */
class NeuralStage(ctx: Context, modelFile: File, inputBytes: Int) : AutoCloseable {
    private val env = Environment.create(BuiltinNpuAcceleratorProvider(ctx))
    private val model: CompiledModel
    private val input: TensorBuffer
    private val outputs: List<List<TensorBuffer>>

    private val freeInputs = ArrayBlockingQueue<ByteArray>(3)
    private val work = ArrayBlockingQueue<ByteArray>(1)
    private val freeSets = ArrayBlockingQueue<Int>(2)
    private val toRead = ArrayBlockingQueue<Int>(2)
    private val latest = AtomicReference<ByteArray?>(null)
    private val finished = AtomicInteger(0)
    private val runThread: Thread
    private val readThread: Thread

    @Volatile private var closed = false
    @Volatile var error: Throwable? = null
        private set
    @Volatile var runMs = 0f
        private set
    @Volatile var readMs = 0f
        private set

    init {
        val options = CompiledModel.Options(Accelerator.NPU).apply {
            qualcommOptions = CompiledModel.QualcommOptions(
                htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
            )
        }
        model = CompiledModel.create(modelFile.absolutePath, options, env)
        input = model.createInputBuffers()[0]
        outputs = listOf(model.createOutputBuffers(), model.createOutputBuffers())
        // The first run on each output set registers its memory with the NPU and can take a second. Pay that now,
        // before the first captured frame, so it is not mistaken for a slow NPU.
        input.writeInt8(ByteArray(inputBytes))
        for (set in outputs) model.run(listOf(input), set)
        repeat(3) { freeInputs.add(ByteArray(inputBytes)) }
        freeSets.add(0)
        freeSets.add(1)
        runThread = Thread({ runLoop() }, "npu-run").also { it.start() }
        readThread = Thread({ readLoop() }, "npu-read").also { it.start() }
    }

    /** Number of results finished so far. */
    fun finishedCount(): Int = finished.get()

    /** GL thread. Copies [packed] in and returns true, or returns false (frame dropped) while the NPU is still busy. */
    fun submit(packed: ByteBuffer): Boolean {
        if (closed || error != null) return false
        val a = freeInputs.poll() ?: return false
        packed.rewind()
        packed.get(a, 0, a.size)
        if (!work.offer(a)) {
            freeInputs.offer(a)
            return false
        }
        return true
    }

    /** GL thread. The newest finished output (planar int8 bytes), or null if nothing new since the last call. */
    fun takeLatest(): ByteArray? = latest.getAndSet(null)

    private fun runLoop() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY)
        try {
            while (!closed) {
                val a = work.take()
                val k = freeSets.take()
                val t0 = System.nanoTime()
                input.writeInt8(a)
                freeInputs.offer(a)
                model.run(listOf(input), outputs[k])
                runMs = (System.nanoTime() - t0) / 1e6f
                toRead.put(k)
            }
        } catch (_: InterruptedException) {
        } catch (t: Throwable) {
            error = t
        }
    }

    private fun readLoop() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY)
        try {
            while (!closed) {
                val k = toRead.take()
                val t0 = System.nanoTime()
                val bytes = outputs[k][0].readInt8()
                readMs = (System.nanoTime() - t0) / 1e6f
                freeSets.offer(k)
                latest.set(bytes)
                finished.incrementAndGet()
            }
        } catch (_: InterruptedException) {
        } catch (t: Throwable) {
            error = t
        }
    }

    override fun close() {
        closed = true
        runThread.interrupt()
        readThread.interrupt()
        runCatching { runThread.join(1000) }
        runCatching { readThread.join(1000) }
        runCatching { model.close() }
        runCatching { (env as? AutoCloseable)?.close() }
    }
}
