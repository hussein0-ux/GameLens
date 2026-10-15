#include <jni.h>
#include <android/hardware_buffer.h>
#include <GLES3/gl30.h>
#include <GLES3/gl3ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <vector>
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnTensor.h"

// Ring buffer sizing to handle simultaneous Read/Compute/Write loops
#define RING_BUFFER_SIZE 3

struct FrameBuffers {
    AHardwareBuffer* inBuffer = nullptr;
    AHardwareBuffer* outBuffer = nullptr;
    EGLImageKHR inImage = EGL_NO_IMAGE_KHR;
    EGLImageKHR outImage = EGL_NO_IMAGE_KHR;
    GLuint inTexture = 0;
    GLuint outTexture = 0;
    
    // QNN Native Memory Wrappers
    Qnn_Tensor_t qnnInputTensor;
    Qnn_Tensor_t qnnOutputTensor;
};

class NativeNpuEngine {
private:
    FrameBuffers ringBuffers[RING_BUFFER_SIZE];
    uint32_t currentFrameIndex = 0;
    
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_ContextHandle_t contextHandle = nullptr;
    Qnn_GraphHandle_t graphHandle = nullptr;
    
    int tileW = 128;
    int tileH = 128;
    int batchSize = 8;

public:
    void initializeBuffers(int width, int height) {
        // Allocate direct graphic hardware buffers matching Npu shape layout
        // The Input buffer contains all 8 tiles packed layout: [8, 128, 128, 3]
        AHardwareBuffer_Desc inDesc = {
            .width = static_cast<uint32_t>(tileW),
            .height = static_cast<uint32_t>(tileH * batchSize), // Stacked vertically in layout
            .layers = 1,
            .format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
            .usage = AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_NEVER
        };

        AHardwareBuffer_Desc outDesc = {
            .width = static_cast<uint32_t>(tileW * 4), 
            .height = static_cast<uint32_t>((tileH * 4) * batchSize), // Upscaled outputs
            .format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
            .usage = AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
        };

        for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
            AHardwareBuffer_allocate(&inDesc, &ringBuffers[i].inBuffer);
            AHardwareBuffer_allocate(&outDesc, &ringBuffers[i].outBuffer);
            
            // Map the buffers directly into OpenGL Client Textures via EGL Extensions
            EGLDisplay dpy = eglGetCurrentDisplay();
            
            ringBuffers[i].inImage = eglCreateImageKHR(dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, 
                                                      reinterpret_cast<EGLClientBuffer>(ringBuffers[i].inBuffer), nullptr);
            ringBuffers[i].outImage = eglCreateImageKHR(dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, 
                                                       reinterpret_cast<EGLClientBuffer>(ringBuffers[i].outBuffer), nullptr);

            glGenTextures(1, &ringBuffers[i].inTexture);
            glBindTexture(GL_TEXTURE_2D, ringBuffers[i].inTexture);
            glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, ringBuffers[i].inImage);

            glGenTextures(1, &ringBuffers[i].outTexture);
            glBindTexture(GL_TEXTURE_2D, ringBuffers[i].outTexture);
            glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, ringBuffers[i].outImage);
            
            // Bind these Hardware Buffers directly into QNN Memory Blocks
            setupQnnUserBuffer(ringBuffers[i]);
        }
    }

    void setupQnnUserBuffer(FrameBuffers& fb) {
        // Map native buffer descriptors to Qnn_ClientBuffer_t
        // Setting memoryType configuration to QNN_TENSOR_MEM_TYPE_ION / QNN_TENSOR_MEM_TYPE_HARDWARE_BUFFER
        fb.qnnInputTensor.v1.clientBuffer.data = reinterpret_cast<void*>(fb.inBuffer);
        fb.qnnInputTensor.v1.clientBuffer.dataSize = tileW * (tileH * batchSize) * 4;
        
        fb.qnnOutputTensor.v1.clientBuffer.data = reinterpret_cast<void*>(fb.outBuffer);
        fb.qnnOutputTensor.v1.clientBuffer.dataSize = (tileW * 4) * ((tileH * 4) * batchSize) * 4;
    }

    void executePipelineFrame(GLuint inputOesTexture, float* stMatrix) {
        uint32_t writeIdx = currentFrameIndex % RING_BUFFER_SIZE;
        uint32_t computeIdx = (currentFrameIndex - 1) % RING_BUFFER_SIZE;
        uint32_t readIdx = (currentFrameIndex - 2) % RING_BUFFER_SIZE;

        // STAGE 1: GPU Downscaling and packing multiple viewport allocations into the Input Buffer
        // This utilizes your existing vertex shaders but shifts viewports to tile structures in one layout pass
        glBindFramebuffer(GL_FRAMEBUFFER, ringBuffers[writeIdx].inTexture);
        for(int t = 0; t < batchSize; ++t) {
            glViewport(0, t * tileH, tileW, tileH);
            // Execute downscale draw logic for each target matrix quadrant location
        }
        glFlush(); // Force execution pipelines to complete operations without CPU wait states

        // STAGE 2: Direct Asynchronous Batch NPU Execution via Native QNN Core Engine
        if (currentFrameIndex > 0) {
            Qnn_Tensor_t inputs[] = { ringBuffers[computeIdx].qnnInputTensor };
            Qnn_Tensor_t outputs[] = { ringBuffers[computeIdx].qnnOutputTensor };
            
            // The Hexagon HTP processes all 8 tiles inside a single transaction sweep
            QnnGraph_executeAsync(graphHandle, inputs, 1, outputs, 1, nullptr, nullptr);
        }

        // STAGE 3: Display output presentation pass sampling from the Upscaled Hardware Buffer Texture
        if (currentFrameIndex > 1) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, screenW, screenH);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ringBuffers[readIdx].outTexture);
            // Draw full viewport layout matching present texturing nodes...
        }

        currentFrameIndex++;
    }
};
