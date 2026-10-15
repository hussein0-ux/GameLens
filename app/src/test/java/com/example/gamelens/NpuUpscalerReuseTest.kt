package com.example.gamelens

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.test.Test
import kotlin.test.assertEquals
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class NpuUpscalerReuseTest {

    @Test
    fun reusedCachedTensorMatchesFreshTensorsElementwise() {
        val modelPath = System.getProperty("gamelens.test.model", "src/test/resources/model.onnx")
        val inputName = "input"
        val shape = longArrayOf(1L, 3L, 4L, 4L)
        val total = shape.fold(1L) { acc, d -> acc * d }.toInt()

        fun freshBuffer(fill: (Int) -> Float): FloatBuffer {
            return ByteBuffer.allocateDirect(total * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .apply { for (i in 0 until total) put(i, fill(i)); rewind() }
        }

        val bufA = freshBuffer { (it % 100 / 100f).toFloat() }
        val bufB = freshBuffer { ((it * 17 + 5) % 100 / 100f).toFloat() }

        val refA = OrtSession.SessionOptions().use { opts ->
            OrtEnvironment.getEnvironment().createSession(modelPath, opts).use { session ->
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), bufA, shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val src = (results[0] as OnnxTensor).floatBuffer
                        FloatBuffer.allocate(src.remaining()).also { it.put(src); it.rewind() }
                    }
                }
            }
        }

        val refB = OrtSession.SessionOptions().use { opts ->
            OrtEnvironment.getEnvironment().createSession(modelPath, opts).use { session ->
                OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), bufB, shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { results ->
                        val src = (results[0] as OnnxTensor).floatBuffer
                        FloatBuffer.allocate(src.remaining()).also { it.put(src); it.rewind() }
                    }
                }
            }
        }

        val env = OrtEnvironment.getEnvironment()
        val scratch = ByteBuffer.allocateDirect(total * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val cached = OnnxTensor.createTensor(env, scratch, shape)
        val session = env.createSession(modelPath, OrtSession.SessionOptions())
        try {
            scratch.clear(); scratch.put(bufA); scratch.rewind()
            val outA = session.run(mapOf(inputName to cached)).use { results ->
                val src = (results[0] as OnnxTensor).floatBuffer
                FloatBuffer.allocate(src.remaining()).also { it.put(src); it.rewind() }
            }

            scratch.clear(); scratch.put(bufB); scratch.rewind()
            val outB = session.run(mapOf(inputName to cached)).use { results ->
                val src = (results[0] as OnnxTensor).floatBuffer
                FloatBuffer.allocate(src.remaining()).also { it.put(src); it.rewind() }
            }

            val tolerance = 1e-5f
            assertEquals(refA.capacity(), outA.capacity(), "Output size mismatch for A")
            assertEquals(refB.capacity(), outB.capacity(), "Output size mismatch for B")
            for (i in 0 until refA.capacity()) {
                assertEquals(refA.get(i), outA.get(i), tolerance, "A mismatch at index $i")
            }
            for (i in 0 until refB.capacity()) {
                assertEquals(refB.get(i), outB.get(i), tolerance, "B mismatch at index $i")
            }
        } finally {
            cached.close()
            session.close()
        }
    }
}
