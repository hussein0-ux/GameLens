package com.example.gamelens
​import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
​class NpuUpscaler(context: Context) {
private var interpreter: Interpreter? = null
private var nnApiDelegate: NnApiDelegate? = null
​companion object {
private const val TAG = "NpuUpscaler"
private const val MODEL_NAME = "quicksrnet_2x_48x4_960x540_w8a8_ft.tflite"
​// Model specific dimensions based on the AI Hub INT8 export
private const val INPUT_WIDTH = 960
private const val INPUT_HEIGHT = 540
private const val OUTPUT_WIDTH = 1920
private const val OUTPUT_HEIGHT = 1080
private const val CHANNELS = 3
​// Quantization parameters (INT8 mapping to 0.0 - 1.0)
private const val SCALE = 0.003922f
private const val ZERO_POINT = -128
}
​// NCHW layout buffers (1 * Channels * Height * Width * 1 byte for INT8)
private val inputBuffer = ByteBuffer.allocateDirect(1 * CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * 1).apply {
order(ByteOrder.nativeOrder())
}
private val outputBuffer = ByteBuffer.allocateDirect(1 * CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH * 1).apply {
order(ByteOrder.nativeOrder())
}
​init {
try {
setupInterpreter(context)
} catch (e: Exception) {
Log.e(TAG, "Failed to initialize TFLite Interpreter: ${e.message}")
}
}
​private fun setupInterpreter(context: Context) {
// Enforce NPU execution via Android NNAPI
nnApiDelegate = NnApiDelegate()
val options = Interpreter.Options().apply {
addDelegate(nnApiDelegate)
setNumThreads(4) // Fallback threads if a specific op isn't NPU-supported
}
​val assetFileDescriptor = context.assets.openFd(MODEL_NAME)
val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
val fileChannel = fileInputStream.channel
val startOffset = assetFileDescriptor.startOffset
val declaredLength = assetFileDescriptor.declaredLength
val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
​interpreter = Interpreter(modelBuffer, options)
Log.i(TAG, "Interpreter initialized successfully with NNAPI delegate.")
}
​/
* Upscales an IntArray of standard ARGB pixels.
* Maps NHWC Android bitmaps to NCHW TFLite tensors.
*/
fun upscale(inputRgb: IntArray): IntArray {
inputBuffer.rewind()
outputBuffer.rewind()
​// 1. Quantize and pack NHWC (ARGB) into NCHW (INT8)
for (c in 0 until CHANNELS) {
for (i in 0 until INPUT_WIDTH * INPUT_HEIGHT) {
val pixel = inputRgb[i]
val channelVal = when(c) {
0 -> (pixel shr 16) and 0xFF // Red
1 -> (pixel shr 8) and 0xFF  // Green
2 -> pixel and 0xFF          // Blue
else -> 0
}
​// Normalize [0, 255] to [0.0, 1.0]
val realPixel = channelVal / 255.0f
// Quantize: q = (real / scale) + zero_point
val quantized = ((realPixel / SCALE) + ZERO_POINT).toInt().coerceIn(-128, 127).toByte()
inputBuffer.put(quantized)
}
}
​// 2. Run Inference
val startTime = System.nanoTime()
interpreter?.run(inputBuffer, outputBuffer)
val endTime = System.nanoTime()
​// Log to verify we are hitting the ~5.9 ms/frame NPU speed target
Log.d(TAG, "NPU Inference time: ${(endTime - startTime) / 1_000_000} ms")
​// 3. Dequantize and unpack NCHW (INT8) back to NHWC (ARGB)
outputBuffer.rewind()
val outputRgb = IntArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
​// Isolate the NCHW channels
val rChannel = ByteArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
val gChannel = ByteArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
val bChannel = ByteArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
​outputBuffer.get(rChannel)
outputBuffer.get(gChannel)
outputBuffer.get(bChannel)
​for (i in 0 until OUTPUT_WIDTH * OUTPUT_HEIGHT) {
// Dequantize: real = (q - zero_point) * scale
val rReal = ((rChannel[i].toInt() - ZERO_POINT) * SCALE).coerceIn(0.0f, 1.0f)
val gReal = ((gChannel[i].toInt() - ZERO_POINT) * SCALE).coerceIn(0.0f, 1.0f)
val bReal = ((bChannel[i].toInt() - ZERO_POINT) * SCALE).coerceIn(0.0f, 1.0f)
​// Map back to standard [0, 255] color space
val r = (rReal * 255).toInt()
val g = (gReal * 255).toInt()
val b = (bReal * 255).toInt()
​// Pack back into ARGB (Alpha = 255)
outputRgb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
}
​return outputRgb
}
​fun close() {
interpreter?.close()
nnApiDelegate?.close()
}
}
