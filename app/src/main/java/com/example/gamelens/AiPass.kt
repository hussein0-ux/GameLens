package com.example.gamelens
​import android.content.Context
import android.util.Log
​class AiPass(context: Context) {
​private val upscaler = NpuUpscaler(context)
​companion object {
private const val TAG = "AiPass"
​// Strict dimensions mapped to the INT8 QuickSRNet model architecture
const val INPUT_WIDTH = 960
const val INPUT_HEIGHT = 540
const val OUTPUT_WIDTH = 1920
const val OUTPUT_HEIGHT = 1080
}
​/
* Processes an incoming frame array through the NPU Pipeline.
* Expected to be called directly by the GlPipeline after reading pixels from the FBO.
*/
fun processFrame(inputPixels: IntArray): IntArray {
// Guard rail to prevent ByteBuffer overflow/underflow crashes
if (inputPixels.size != INPUT_WIDTH * INPUT_HEIGHT) {
Log.e(TAG, "Frame size mismatch. Expected {INPUT_WIDTH * INPUT_HEIGHT}, got{inputPixels.size}")
// Return an empty/black frame gracefully to avoid crashing the render thread
return IntArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
}
​return upscaler.upscale(inputPixels)
}
​fun release() {
upscaler.close()
}
}
