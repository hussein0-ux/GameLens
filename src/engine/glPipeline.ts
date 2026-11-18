import type { CubeLut } from './cubeLut';
import {
  BLIT_FRAGMENT_SRC,
  BLIT_VERTEX_SRC,
  COPY_FRAGMENT_SRC,
  GSR_COPY_FRAGMENT_SRC,
  GSR_COPY_VERTEX_SRC,
  GSR_VERTEX_SRC,
  LOOK_FRAGMENT_SRC,
  SGSR1_FRAGMENT_SRC,
  SPLIT_FRAGMENT_SRC,
  UPSCALE_FRAGMENT_SRC,
  VERTEX_SRC,
} from './glShaders';
import {
  MODE_AUTO,
  MODE_GPU,
  MODE_NEURAL,
  UPSCALER_GSR,
  type LookSettingsState,
} from './settings';
import { NeuralGovernor } from './neuralGovernor';
import type { NeuralModelInfo } from './tfliteParser';

export interface PipelineDiagnostics {
  captureW: number;
  captureH: number;
  outputW: number;
  outputH: number;
  capturedFps: number;
  drawnFps: number;
  frameMs: number;
  activePath: string;
  lutLoaded: boolean;
  lutSize: number;
  aiStatus: string;
  aiStats: string;
  diagText: string;
}

export class WebGlPipeline {
  private gl: WebGL2RenderingContext;
  private quadVao: WebGLVertexArrayObject;
  private quadBuf: WebGLBuffer;

  private sourceTex: WebGLTexture;
  private capFbo: WebGLFramebuffer;
  private capTex: WebGLTexture;
  private capW = 0;
  private capH = 0;

  private outFbo: WebGLFramebuffer;
  private outTex: WebGLTexture;
  private outW = 0;
  private outH = 0;

  // Motion blur ping-pong FBOs
  private motionFbo: WebGLFramebuffer;
  private motionTex: [WebGLTexture, WebGLTexture];
  private motionCur = 0;
  private hasPrev = false;
  private motionW = 0;
  private motionH = 0;

  // GSR intermediate 2D FBO
  private gsrFbo: WebGLFramebuffer;
  private gsrTex: WebGLTexture;
  private gsrW = 0;
  private gsrH = 0;

  // 3D LUT
  private lutTex: WebGLTexture | null = null;
  private lutSize = 0;

  // Programs
  private copyProg: WebGLProgram;
  private lookProg: WebGLProgram;
  private upProg: WebGLProgram;
  private gsrCopyProg: WebGLProgram;
  private gsrProg: WebGLProgram;
  private blitProg: WebGLProgram;
  private splitProg: WebGLProgram;

  private programsToDelete: WebGLProgram[] = [];
  private governor = new NeuralGovernor();
  private govSince = performance.now();
  private govStart = performance.now();
  private lastMode = MODE_GPU;

  private dgFrames = 0;
  private dgSince = performance.now();
  private dgFrameMsAcc = 0;
  private lastDiag: PipelineDiagnostics = {
    captureW: 0,
    captureH: 0,
    outputW: 0,
    outputH: 0,
    capturedFps: 0,
    drawnFps: 0,
    frameMs: 0,
    activePath: 'Initializing',
    lutLoaded: false,
    lutSize: 0,
    aiStatus: 'Neural idle',
    aiStats: '',
    diagText: '',
  };

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      throw new Error('WebGL2 is not supported in this browser');
    }
    this.gl = gl;

    // Quad geometry: x, y, u, v
    const quadData = new Float32Array([
      -1, -1, 0, 0,
       1, -1, 1, 0,
      -1,  1, 0, 1,
       1,  1, 1, 1,
    ]);
    const vao = gl.createVertexArray();
    const buf = gl.createBuffer();
    if (!vao || !buf) throw new Error('Failed to create WebGL2 buffers');
    this.quadVao = vao;
    this.quadBuf = buf;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, quadData, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);

    this.sourceTex = this.createTex2D(gl.LINEAR);
    this.capTex = this.createTex2D(gl.LINEAR);
    this.capFbo = this.createFbo();
    this.outTex = this.createTex2D(gl.LINEAR);
    this.outFbo = this.createFbo();
    this.gsrTex = this.createTex2D(gl.LINEAR);
    this.gsrFbo = this.createFbo();
    this.motionFbo = this.createFbo();
    this.motionTex = [this.createTex2D(gl.LINEAR), this.createTex2D(gl.LINEAR)];

    this.copyProg = this.buildProgram(VERTEX_SRC, COPY_FRAGMENT_SRC);
    this.lookProg = this.buildProgram(VERTEX_SRC, LOOK_FRAGMENT_SRC);
    this.upProg = this.buildProgram(VERTEX_SRC, UPSCALE_FRAGMENT_SRC);
    this.gsrCopyProg = this.buildProgram(GSR_COPY_VERTEX_SRC, GSR_COPY_FRAGMENT_SRC);
    this.gsrProg = this.buildProgram(GSR_VERTEX_SRC, SGSR1_FRAGMENT_SRC);
    this.blitProg = this.buildProgram(BLIT_VERTEX_SRC, BLIT_FRAGMENT_SRC);
    this.splitProg = this.buildProgram(BLIT_VERTEX_SRC, SPLIT_FRAGMENT_SRC);
  }

  private createFbo(): WebGLFramebuffer {
    const f = this.gl.createFramebuffer();
    if (!f) throw new Error('Failed to create WebGL framebuffer');
    return f;
  }

  private createTex2D(filter: number): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture();
    if (!t) throw new Error('Failed to create WebGL texture');
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private buildProgram(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type);
      if (!s) throw new Error('Unable to create shader');
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s) || 'Unknown shader compile error';
        gl.deleteShader(s);
        throw new Error(log);
      }
      return s;
    };

    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    if (!p) throw new Error('Unable to create program');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p) || 'Unknown link error';
      gl.deleteProgram(p);
      throw new Error(log);
    }
    this.programsToDelete.push(p);
    return p;
  }

  uploadLut(lut: CubeLut): void {
    const gl = this.gl;
    if (this.lutTex) {
      gl.deleteTexture(this.lutTex);
      this.lutTex = null;
    }
    const tex = gl.createTexture();
    if (!tex) return;

    const total = lut.size * lut.size * lut.size;
    const floatLinear = gl.getExtension('OES_texture_float_linear');

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    if (floatLinear) {
      // RGBA32F with OES_texture_float_linear gives full precision 3D trilinear interpolation
      const rgba = new Float32Array(total * 4);
      for (let i = 0; i < total; i++) {
        rgba[i * 4] = lut.rgb[i * 3];
        rgba[i * 4 + 1] = lut.rgb[i * 3 + 1];
        rgba[i * 4 + 2] = lut.rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0;
      }
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        gl.RGBA32F,
        lut.size,
        lut.size,
        lut.size,
        0,
        gl.RGBA,
        gl.FLOAT,
        rgba,
      );
    } else {
      // Universal WebGL2 fallback: RGBA8 is guaranteed linear-filterable everywhere
      const rgba = new Uint8Array(total * 4);
      for (let i = 0; i < total; i++) {
        rgba[i * 4] = Math.round(Math.min(Math.max(lut.rgb[i * 3], 0), 1) * 255);
        rgba[i * 4 + 1] = Math.round(Math.min(Math.max(lut.rgb[i * 3 + 1], 0), 1) * 255);
        rgba[i * 4 + 2] = Math.round(Math.min(Math.max(lut.rgb[i * 3 + 2], 0), 1) * 255);
        rgba[i * 4 + 3] = 255;
      }
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        gl.RGBA8,
        lut.size,
        lut.size,
        lut.size,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        rgba,
      );
    }

    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.activeTexture(gl.TEXTURE0);

    this.lutTex = tex;
    this.lutSize = lut.size;
    this.lastDiag.lutLoaded = true;
    this.lastDiag.lutSize = lut.size;
  }

  private ensureTarget2D(
    fbo: WebGLFramebuffer,
    tex: WebGLTexture,
    w: number,
    h: number,
    curW: number,
    curH: number,
  ): boolean {
    if (curW === w && curH === h) return false;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return true;
  }

  private ensureMotionTargets(w: number, h: number): void {
    if (this.motionW === w && this.motionH === h) return;
    const gl = this.gl;
    for (let i = 0; i < 2; i++) {
      gl.bindTexture(gl.TEXTURE_2D, this.motionTex[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    this.motionW = w;
    this.motionH = h;
    this.motionCur = 0;
    this.hasPrev = false;
  }

  private drawQuad(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  renderFrame(
    source: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
    srcW: number,
    srcH: number,
    outW: number,
    outH: number,
    settings: LookSettingsState,
    neuralModel: NeuralModelInfo | null,
    splitOpts: { enabled: boolean; x: number; zoom: number; zoomCenter: [number, number] },
  ): PipelineDiagnostics {
    const t0 = performance.now();
    const gl = this.gl;

    // 1) Upload source image/video frame into sourceTex
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);

    // 2) Capture resolution (half screen size when captureHalf is enabled)
    const capW = settings.captureHalf ? Math.max(64, Math.floor(srcW / 2)) : srcW;
    const capH = settings.captureHalf ? Math.max(64, Math.floor(srcH / 2)) : srcH;
    if (this.ensureTarget2D(this.capFbo, this.capTex, capW, capH, this.capW, this.capH)) {
      this.capW = capW;
      this.capH = capH;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.capFbo);
    gl.viewport(0, 0, capW, capH);
    gl.useProgram(this.copyProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTex);
    gl.uniform1i(gl.getUniformLocation(this.copyProg, 'uTex'), 0);
    gl.uniform1f(gl.getUniformLocation(this.copyProg, 'uFlipY'), 0.0);
    this.drawQuad();

    // Ensure final output FBO matches outW x outH
    if (this.ensureTarget2D(this.outFbo, this.outTex, outW, outH, this.outW, this.outH)) {
      this.outW = outW;
      this.outH = outH;
    }

    // Track neural / governor status (faithful to GlPipeline.kt & NeuralGovernor.kt)
    const mode = settings.mode;
    if (mode !== this.lastMode && mode >= MODE_NEURAL) {
      this.governor.reset();
      this.govStart = performance.now();
      this.govSince = this.govStart;
    }
    this.lastMode = mode;

    let aiStatus = 'Neural idle (not selected)';
    let aiStats = '';
    if (mode >= MODE_NEURAL) {
      if (!neuralModel) {
        aiStatus = 'Neural: pick a .tflite model in the app first (showing GPU fallback)';
      } else if (Math.abs(capW - neuralModel.w) > 8 || Math.abs(capH - neuralModel.h) > 8) {
        aiStatus = `Neural waiting: capture is ${capW}x${capH}, model wants ${neuralModel.w}x${neuralModel.h}. Showing GPU.`;
        aiStats = `Model ${neuralModel.fileName}: ${neuralModel.w}x${neuralModel.h} -> ${neuralModel.w * neuralModel.scale}x${neuralModel.h * neuralModel.scale} (int8 NCHW)`;
      } else {
        aiStatus =
          mode === MODE_AUTO && !this.governor.useNeural
            ? 'Neural fallback -> GPU (Auto fell back below 110 fps)'
            : 'Neural NPU delegate unavailable in browser -> GPU upscale active';
        aiStats = `Model ${neuralModel.fileName}: ${neuralModel.w}x${neuralModel.h} x${neuralModel.scale} (inScale=${neuralModel.inScale.toFixed(5)}, outScale=${neuralModel.outScale.toFixed(5)})`;
      }
    }

    const lutAmt = this.lutTex ? settings.lut : 0;
    const lutMapX = this.lutSize > 0 ? (this.lutSize - 1) / this.lutSize : 1;
    const lutMapY = this.lutSize > 0 ? 0.5 / this.lutSize : 0;

    let activePath = 'Bypass (Raw)';

    if (settings.bypass) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
      gl.viewport(0, 0, outW, outH);
      gl.useProgram(this.copyProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.capTex);
      gl.uniform1i(gl.getUniformLocation(this.copyProg, 'uTex'), 0);
      gl.uniform1f(gl.getUniformLocation(this.copyProg, 'uFlipY'), 0.0);
      this.drawQuad();
      activePath = 'Bypass (Raw Capture)';
    } else if (mode >= MODE_GPU) {
      if (settings.upscaler === UPSCALER_GSR) {
        // Qualcomm GSR path: capTex -> gsrCopy (FXAA 3 Console) into gsrTex -> SGSR1 + LUT into outFbo
        if (this.ensureTarget2D(this.gsrFbo, this.gsrTex, capW, capH, this.gsrW, this.gsrH)) {
          this.gsrW = capW;
          this.gsrH = capH;
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.gsrFbo);
        gl.viewport(0, 0, capW, capH);
        gl.useProgram(this.gsrCopyProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.capTex);
        gl.uniform1i(gl.getUniformLocation(this.gsrCopyProg, 'uTex'), 0);
        gl.uniform2f(gl.getUniformLocation(this.gsrCopyProg, 'uTexel'), 1 / capW, 1 / capH);
        gl.uniform1f(gl.getUniformLocation(this.gsrCopyProg, 'uAa'), settings.aa);
        this.drawQuad();

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
        gl.viewport(0, 0, outW, outH);
        gl.useProgram(this.gsrProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.gsrTex);
        gl.uniform1i(gl.getUniformLocation(this.gsrProg, 'ps0'), 0);
        gl.uniform4f(
          gl.getUniformLocation(this.gsrProg, 'ViewportInfo[0]'),
          1 / capW,
          1 / capH,
          capW,
          capH,
        );
        gl.uniform1f(gl.getUniformLocation(this.gsrProg, 'uEdgeSharpness'), 1.0 + settings.upSharp);
        gl.uniform1f(gl.getUniformLocation(this.gsrProg, 'uClarity'), settings.upClarity);
        gl.uniform1f(gl.getUniformLocation(this.gsrProg, 'uVibrance'), settings.upVibrance);
        gl.uniform1i(gl.getUniformLocation(this.gsrProg, 'uLut'), 2);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1f(gl.getUniformLocation(this.gsrProg, 'uLutAmt'), lutAmt);
        gl.uniform2f(gl.getUniformLocation(this.gsrProg, 'uLutMap'), lutMapX, lutMapY);
        this.drawQuad();
        activePath = mode >= MODE_NEURAL ? 'Qualcomm GSR (Neural Fallback) + LUT' : 'Qualcomm GSR 1 + LUT';
      } else {
        // Catmull-Rom 2x + CAS sharpen + LUT
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
        gl.viewport(0, 0, outW, outH);
        gl.useProgram(this.upProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.capTex);
        gl.uniform1i(gl.getUniformLocation(this.upProg, 'uTex'), 0);
        gl.uniform2f(gl.getUniformLocation(this.upProg, 'uTexel'), 1 / capW, 1 / capH);
        gl.uniform1f(gl.getUniformLocation(this.upProg, 'uSharp'), settings.upSharp);
        gl.uniform1i(gl.getUniformLocation(this.upProg, 'uLut'), 2);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
        gl.activeTexture(gl.TEXTURE0);
        gl.uniform1f(gl.getUniformLocation(this.upProg, 'uLutAmt'), lutAmt);
        gl.uniform2f(gl.getUniformLocation(this.upProg, 'uLutMap'), lutMapX, lutMapY);
        gl.uniform1f(gl.getUniformLocation(this.upProg, 'uFlipY'), 0.0);
        this.drawQuad();
        activePath = mode >= MODE_NEURAL ? 'Catmull-Rom 2x (Neural Fallback) + LUT' : 'Catmull-Rom 2x + LUT';
      }
    } else {
      // MODE_LOOK: Heavy single-pass look filters + optional motion blur ping-pong
      const useMotion = settings.motion > 0.01;
      if (useMotion) {
        this.ensureMotionTargets(outW, outH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.motionFbo);
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          this.motionTex[this.motionCur],
          0,
        );
      } else {
        this.hasPrev = false;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
      }

      gl.viewport(0, 0, outW, outH);
      gl.useProgram(this.lookProg);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.motionTex[1 - this.motionCur]);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.capTex);

      gl.uniform1i(gl.getUniformLocation(this.lookProg, 'uTex'), 0);
      gl.uniform1i(gl.getUniformLocation(this.lookProg, 'uPrev'), 1);
      gl.uniform1i(gl.getUniformLocation(this.lookProg, 'uLut'), 2);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uFlipY'), 0.0);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uBypass'), 0.0);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uLutAmt'), lutAmt);
      gl.uniform2f(gl.getUniformLocation(this.lookProg, 'uLutMap'), lutMapX, lutMapY);
      gl.uniform2f(gl.getUniformLocation(this.lookProg, 'uRes'), outW, outH);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uHasPrev'), useMotion && this.hasPrev ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uMotion'), useMotion ? settings.motion : 0);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uVig'), settings.vignette);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uChroma'), settings.chroma);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uHdr'), settings.hdr);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uBlack'), settings.blackPt);
      gl.uniform2f(gl.getUniformLocation(this.lookProg, 'uTexel'), 1 / capW, 1 / capH);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uSharp'), settings.sharpen);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uClarity'), settings.clarity);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uContrast'), settings.contrast);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uSat'), settings.saturation);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uFilmic'), settings.filmic);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uExposure'), settings.exposure);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uGrade'), settings.grade);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uVibrance'), settings.vibrance);
      gl.uniform1f(gl.getUniformLocation(this.lookProg, 'uGlow'), settings.glow);
      this.drawQuad();

      if (useMotion) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFbo);
        gl.viewport(0, 0, outW, outH);
        gl.useProgram(this.blitProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.motionTex[this.motionCur]);
        gl.uniform1i(gl.getUniformLocation(this.blitProg, 'uTex2'), 0);
        gl.uniform2f(gl.getUniformLocation(this.blitProg, 'uRes'), outW, outH);
        this.drawQuad();
        this.motionCur = 1 - this.motionCur;
        this.hasPrev = true;
      }
      activePath = 'Look Filters + HyperReal LUT';
    }

    // 3) Present final output (with optional before/after split wipe & zoom loupe) to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.splitProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sourceTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.outTex);
    gl.uniform1i(gl.getUniformLocation(this.splitProg, 'uRawTex'), 0);
    gl.uniform1i(gl.getUniformLocation(this.splitProg, 'uOutTex'), 1);
    gl.uniform2f(gl.getUniformLocation(this.splitProg, 'uRes'), outW, outH);
    gl.uniform1f(gl.getUniformLocation(this.splitProg, 'uSplitX'), splitOpts.x);
    gl.uniform1f(gl.getUniformLocation(this.splitProg, 'uSplitEnabled'), splitOpts.enabled ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.splitProg, 'uZoom'), splitOpts.zoom);
    gl.uniform2f(
      gl.getUniformLocation(this.splitProg, 'uZoomCenter'),
      splitOpts.zoomCenter[0],
      splitOpts.zoomCenter[1],
    );
    this.drawQuad();
    gl.activeTexture(gl.TEXTURE0);

    const elapsedMs = performance.now() - t0;
    this.dgFrames++;
    this.dgFrameMsAcc += elapsedMs;

    const now = performance.now();
    const dtSec = (now - this.dgSince) / 1000;
    if (dtSec >= 0.5) {
      const fps = this.dgFrames / dtSec;
      const avgMs = this.dgFrameMsAcc / this.dgFrames;
      if (mode === MODE_AUTO && now - this.govStart > 3000 && now - this.govSince >= 1000) {
        this.governor.onSecond(fps);
        this.govSince = now;
      }
      this.lastDiag = {
        captureW: capW,
        captureH: capH,
        outputW: outW,
        outputH: outH,
        capturedFps: fps,
        drawnFps: fps,
        frameMs: avgMs,
        activePath,
        lutLoaded: Boolean(this.lutTex),
        lutSize: this.lutSize,
        aiStatus,
        aiStats,
        diagText: `capture ${fps.toFixed(0)} frames/s, drawn ${fps.toFixed(0)}/s, render ${avgMs.toFixed(2)} ms/frame`,
      };
      this.dgFrames = 0;
      this.dgFrameMsAcc = 0;
      this.dgSince = now;
    } else {
      this.lastDiag.captureW = capW;
      this.lastDiag.captureH = capH;
      this.lastDiag.outputW = outW;
      this.lastDiag.outputH = outH;
      this.lastDiag.activePath = activePath;
      this.lastDiag.aiStatus = aiStatus;
      this.lastDiag.aiStats = aiStats;
    }

    return { ...this.lastDiag };
  }

  release(): void {
    const gl = this.gl;
    for (const p of this.programsToDelete) {
      gl.deleteProgram(p);
    }
    gl.deleteTexture(this.sourceTex);
    gl.deleteTexture(this.capTex);
    gl.deleteTexture(this.outTex);
    gl.deleteTexture(this.gsrTex);
    gl.deleteTexture(this.motionTex[0]);
    gl.deleteTexture(this.motionTex[1]);
    if (this.lutTex) gl.deleteTexture(this.lutTex);
    gl.deleteFramebuffer(this.capFbo);
    gl.deleteFramebuffer(this.outFbo);
    gl.deleteFramebuffer(this.gsrFbo);
    gl.deleteFramebuffer(this.motionFbo);
    gl.deleteBuffer(this.quadBuf);
    gl.deleteVertexArray(this.quadVao);
  }
}
