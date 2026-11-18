import { DlssTier } from '../core/dlss5Math';
import { parseCube } from '../core/cubeLut';
import {
  LookState,
  MODE_AUTO,
  MODE_DLSS5,
  MODE_GPU,
  MODE_NEURAL,
  UPSCALER_GSR,
} from '../core/lookSettings';
import { NeuralGovernor } from '../core/neuralGovernor';
import { NeuralModelInfo } from '../core/tfliteModel';
import { recordCrash } from '../core/crashLog';
import { Dlss5Engine } from './Dlss5Engine';
import {
  BLIT_FRAGMENT_SRC,
  BLIT_VERTEX_SRC,
  GSR_COPY_FRAGMENT_SRC,
  GSR_COPY_VERTEX_SRC,
  GSR_FRAGMENT_SRC,
  GSR_VERTEX_SRC,
  LOOK_FRAGMENT_SRC,
  UPSCALE_FRAGMENT_SRC,
  VERTEX_SRC,
} from './shaders';

export interface PipelineTelemetry {
  status: string;
  stats: string;
  diag: string;
  captureWidth: number;
  captureHeight: number;
  outputWidth: number;
  outputHeight: number;
  lutLoaded: boolean;
  lutSize: number;
  gsrAvailable: boolean;
  dlss5Available: boolean;
  dlss5Tier: DlssTier;
  dlss5GbufMs: number;
  dlss5ReconMs: number;
  dlss5FrameGen: boolean;
}

export class WebGlPipeline {
  private readonly gl: WebGL2RenderingContext;
  private quadVao: WebGLVertexArrayObject | null = null;
  private quadBuf: WebGLBuffer | null = null;

  private srcTex: WebGLTexture | null = null;
  private halfTex: WebGLTexture | null = null;
  private halfFbo: WebGLFramebuffer | null = null;
  private halfW = 0;
  private halfH = 0;

  private lutTex: WebGLTexture | null = null;
  private lutSize = 0;

  // Programs
  private lookProg: WebGLProgram | null = null;
  private upProg: WebGLProgram | null = null;
  private gsrCopyProg: WebGLProgram | null = null;
  private gsrProg: WebGLProgram | null = null;
  private blitProg: WebGLProgram | null = null;

  // GSR intermediate target
  private gsrTex: WebGLTexture | null = null;
  private gsrFbo: WebGLFramebuffer | null = null;
  private gsrW = 0;
  private gsrH = 0;

  // Motion blur ping-pong FBOs
  private fbo: WebGLFramebuffer | null = null;
  private fboTex: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  private fboW = 0;
  private fboH = 0;
  private cur = 0;
  private hasPrev = false;

  // Uniforms cache
  private uLook: Record<string, WebGLUniformLocation | null> = {};
  private uUp: Record<string, WebGLUniformLocation | null> = {};
  private uGsrCopy: Record<string, WebGLUniformLocation | null> = {};
  private uGsr: Record<string, WebGLUniformLocation | null> = {};
  private uBlit: Record<string, WebGLUniformLocation | null> = {};

  // Diagnostics & NeuralGovernor
  private dgCaptured = 0;
  private dgDrawn = 0;
  private dgDrawMsAcc = 0;
  private dgSince = performance.now();
  private governor = new NeuralGovernor();
  private govSince = performance.now();
  private govStart = performance.now();
  private lastMode = MODE_GPU;
  private released = false;
  private dlss5Engine: Dlss5Engine | null = null;

  public telemetry: PipelineTelemetry = {
    status: 'Neural idle',
    stats: '',
    diag: '',
    captureWidth: 0,
    captureHeight: 0,
    outputWidth: 0,
    outputHeight: 0,
    lutLoaded: false,
    lutSize: 0,
    gsrAvailable: true,
    dlss5Available: false,
    dlss5Tier: 1,
    dlss5GbufMs: 0,
    dlss5ReconMs: 0,
    dlss5FrameGen: false,
  };

  constructor(private readonly canvas: HTMLCanvasElement) {
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
    this.initGl();
  }

  private compileShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error('Failed to create shader');
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${log}`);
    }
    return shader;
  }

  private buildProgram(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, vsSrc);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    if (!prog) throw new Error('Failed to create GL program');
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) || 'Unknown program link error';
      gl.deleteProgram(prog);
      throw new Error(`Program link error: ${log}`);
    }
    return prog;
  }

  private initGl(): void {
    const gl = this.gl;

    // Fullscreen quad: x, y, u, v
    const quadData = new Float32Array([
      -1, -1, 0, 0,
       1, -1, 1, 0,
      -1,  1, 0, 1,
       1,  1, 1, 1,
    ]);
    this.quadVao = gl.createVertexArray();
    this.quadBuf = gl.createBuffer();
    gl.bindVertexArray(this.quadVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quadData, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);

    // Source texture
    this.srcTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Look program
    this.lookProg = this.buildProgram(VERTEX_SRC, LOOK_FRAGMENT_SRC);
    for (const name of [
      'uTex', 'uTexel', 'uSharp', 'uClarity', 'uContrast', 'uSat', 'uFilmic',
      'uExposure', 'uGrade', 'uVibrance', 'uGlow', 'uPrev', 'uRes', 'uHasPrev',
      'uMotion', 'uVig', 'uChroma', 'uHdr', 'uBlack', 'uMicroRelief', 'uDehaze',
      'uShadowLift', 'uBypass', 'uLut', 'uLutAmt', 'uLutMap',
    ]) {
      this.uLook[name] = gl.getUniformLocation(this.lookProg, name);
    }

    // Catmull-Rom upscale program
    this.upProg = this.buildProgram(VERTEX_SRC, UPSCALE_FRAGMENT_SRC);
    for (const name of [
      'uTex', 'uTexel', 'uSharp', 'uMicroRelief', 'uDehaze', 'uShadowLift',
      'uHdr', 'uFilmic', 'uExposure', 'uContrast', 'uSat', 'uGrade', 'uBlack',
      'uLut', 'uLutAmt', 'uLutMap',
    ]) {
      this.uUp[name] = gl.getUniformLocation(this.upProg, name);
    }

    // GSR copy + GSR1 upscale programs (optional: falls back to Catmull-Rom on any driver issue, matching Android initGsr)
    try {
      this.gsrCopyProg = this.buildProgram(GSR_COPY_VERTEX_SRC, GSR_COPY_FRAGMENT_SRC);
      for (const name of ['uTex', 'uTexel', 'uAa']) {
        this.uGsrCopy[name] = gl.getUniformLocation(this.gsrCopyProg, name);
      }

      this.gsrProg = this.buildProgram(GSR_VERTEX_SRC, GSR_FRAGMENT_SRC);
      for (const name of [
        'ps0', 'ViewportInfo', 'uEdgeSharpness', 'uClarity', 'uVibrance',
        'uMicroRelief', 'uDehaze', 'uShadowLift', 'uHdr', 'uFilmic', 'uGlow',
        'uExposure', 'uContrast', 'uSat', 'uGrade', 'uBlack', 'uVig',
        'uLut', 'uLutAmt', 'uLutMap',
      ]) {
        this.uGsr[name] = gl.getUniformLocation(this.gsrProg, name);
      }
      this.telemetry.gsrAvailable = true;
    } catch (err) {
      if (this.gsrCopyProg) gl.deleteProgram(this.gsrCopyProg);
      if (this.gsrProg) gl.deleteProgram(this.gsrProg);
      this.gsrCopyProg = null;
      this.gsrProg = null;
      this.telemetry.gsrAvailable = false;
      const msg = err instanceof Error ? err.message : String(err);
      this.telemetry.status = `GSR unavailable (${msg}), using Catmull-Rom`;
    }

    // Blit program
    this.blitProg = this.buildProgram(BLIT_VERTEX_SRC, BLIT_FRAGMENT_SRC);
    for (const name of ['uTex2', 'uRes']) {
      this.uBlit[name] = gl.getUniformLocation(this.blitProg, name);
    }

    // DLSS-5 equivalent 3D G-Buffer + Neural TAAU + Frame Gen engine (optional fallback on driver issues)
    try {
      this.dlss5Engine = new Dlss5Engine(
        gl,
        (vs, fs) => this.buildProgram(vs, fs),
        () => this.drawQuad(),
      );
      this.telemetry.dlss5Available = true;
    } catch (err) {
      recordCrash('Dlss5Engine.init', err);
      this.dlss5Engine = null;
      this.telemetry.dlss5Available = false;
    }
  }

  async loadCubeLut(url: string): Promise<void> {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      if (this.released) return;
      const lut = parseCube(text);
      const gl = this.gl;

      // Pack size^3 RGB floats into RGBA16F-compatible Float32 RGBA buffer
      const total = lut.size * lut.size * lut.size;
      const rgba = new Float32Array(total * 4);
      for (let i = 0; i < total; i++) {
        rgba[i * 4] = lut.rgb[i * 3];
        rgba[i * 4 + 1] = lut.rgb[i * 3 + 1];
        rgba[i * 4 + 2] = lut.rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 1.0;
      }

      if (this.lutTex) gl.deleteTexture(this.lutTex);
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
      // Clear prior GL errors, try RGBA16F first, fall back to RGBA8 if driver rejects half-float 3D LUT
      while (gl.getError() !== gl.NO_ERROR) {
        // flush
      }
      gl.texImage3D(
        gl.TEXTURE_3D,
        0,
        gl.RGBA16F,
        lut.size,
        lut.size,
        lut.size,
        0,
        gl.RGBA,
        gl.FLOAT,
        rgba,
      );
      if (gl.getError() !== gl.NO_ERROR) {
        const rgba8 = new Uint8Array(total * 4);
        for (let i = 0; i < total * 4; i++) {
          rgba8[i] = Math.max(0, Math.min(255, Math.round(rgba[i] * 255)));
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
          rgba8,
        );
        if (gl.getError() !== gl.NO_ERROR) {
          gl.deleteTexture(tex);
          gl.activeTexture(gl.TEXTURE0);
          throw new Error('Failed to upload 3D LUT texture');
        }
      }
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.activeTexture(gl.TEXTURE0);

      this.lutTex = tex;
      this.lutSize = lut.size;
      this.telemetry.lutLoaded = true;
      this.telemetry.lutSize = lut.size;
    } catch (err) {
      recordCrash('loadCubeLut', err);
      this.lutTex = null;
      this.lutSize = 0;
      this.telemetry.lutLoaded = false;
    }
  }

  private ensureHalfTarget(w: number, h: number): void {
    const gl = this.gl;
    if (this.halfW === w && this.halfH === h && this.halfTex && this.halfFbo) return;
    if (this.halfTex) gl.deleteTexture(this.halfTex);
    if (this.halfFbo) gl.deleteFramebuffer(this.halfFbo);
    this.halfW = w;
    this.halfH = h;
    this.halfTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.halfTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.halfFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.halfFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.halfTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private ensureGsrTarget(w: number, h: number): void {
    const gl = this.gl;
    if (this.gsrW === w && this.gsrH === h && this.gsrTex && this.gsrFbo) return;
    if (this.gsrTex) gl.deleteTexture(this.gsrTex);
    if (this.gsrFbo) gl.deleteFramebuffer(this.gsrFbo);
    this.gsrW = w;
    this.gsrH = h;
    this.gsrTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.gsrTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.gsrFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gsrFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.gsrTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private ensureMotionFbo(w: number, h: number): void {
    const gl = this.gl;
    if (this.fboW === w && this.fboH === h && this.fboTex[0] && this.fbo) return;
    if (this.fboTex[0]) gl.deleteTexture(this.fboTex[0]);
    if (this.fboTex[1]) gl.deleteTexture(this.fboTex[1]);
    if (!this.fbo) this.fbo = gl.createFramebuffer();
    for (let i = 0; i < 2; i++) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.fboTex[i] = t;
    }
    this.fboW = w;
    this.fboH = h;
    this.cur = 0;
    this.hasPrev = false;
  }

  private drawQuad(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  private bindLut(amtLoc: WebGLUniformLocation | null, mapLoc: WebGLUniformLocation | null, samplerLoc: WebGLUniformLocation | null, lutStrength: number): void {
    const gl = this.gl;
    gl.uniform1i(samplerLoc, 2);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(amtLoc, this.lutTex ? lutStrength : 0);
    if (this.lutSize > 0) {
      gl.uniform2f(mapLoc, (this.lutSize - 1) / this.lutSize, 0.5 / this.lutSize);
    }
  }

  drawFrame(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    srcW: number,
    srcH: number,
    settings: LookState,
    neuralModel: NeuralModelInfo | null,
  ): void {
    if (this.released || srcW <= 0 || srcH <= 0) return;

    const tStart = performance.now();
    const gl = this.gl;
    this.dgCaptured++;

    // Upload current video/image frame
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);

    const outW = this.canvas.width;
    const outH = this.canvas.height;
    let capW = srcW;
    let capH = srcH;
    let activeInputTex = this.srcTex!;

    // Capture at half screen size simulation when captureHalf is enabled (matches CaptureService.captureSize)
    if (settings.captureHalf && !settings.bypass && settings.mode >= MODE_GPU) {
      capW = Math.max(16, Math.floor(srcW / 2));
      capH = Math.max(16, Math.floor(srcH / 2));
      this.ensureHalfTarget(capW, capH);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.halfFbo);
      gl.viewport(0, 0, capW, capH);
      gl.useProgram(this.blitProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
      gl.uniform1i(this.uBlit.uTex2, 0);
      gl.uniform2f(this.uBlit.uRes, capW, capH);
      this.drawQuad();
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      activeInputTex = this.halfTex!;
    }

    this.telemetry.captureWidth = capW;
    this.telemetry.captureHeight = capH;
    this.telemetry.outputWidth = outW;
    this.telemetry.outputHeight = outH;

    // Neural / Auto status & governor handling (browser does not expose Qualcomm Hexagon NPU;
    // validates the .tflite model dimensions and falls back to GPU upscale via NeuralGovernor)
    const mode = settings.mode;
    if (mode < MODE_NEURAL && this.lastMode >= MODE_NEURAL) {
      this.telemetry.status = 'Neural idle (not selected)';
      this.telemetry.stats = '';
      this.governor = new NeuralGovernor();
      this.govStart = tStart;
      this.govSince = tStart;
    } else if (!settings.bypass && mode >= MODE_NEURAL) {
      if (!neuralModel) {
        this.telemetry.status = 'Neural: pick a .tflite model in the app first';
        this.telemetry.stats = '';
      } else if (Math.abs(capW - neuralModel.w) > 8 || Math.abs(capH - neuralModel.h) > 8) {
        this.telemetry.status = `Neural waiting: capture is ${capW}x${capH}, the model wants ${neuralModel.w}x${neuralModel.h}. Showing GPU.`;
        this.telemetry.stats = '';
      } else {
        // Browser lacks direct Hexagon NPU FastRPC access; feed 0 neural results/s into NeuralGovernor in Auto mode.
        const dtGov = (tStart - this.govSince) / 1000;
        if (dtGov >= 1.0) {
          this.govSince = tStart;
          if (tStart - this.govStart > 3000) {
            this.governor.onSecond(0);
          }
        }
        const showing =
          mode === MODE_AUTO && !this.governor.useNeural
            ? 'GPU (Auto fell back)'
            : 'GPU (Hexagon NPU requires Android device)';
        this.telemetry.status = `Neural 0 results/s, NPU -- ms, read -- ms, showing ${showing}`;
        this.telemetry.stats = `Model ${neuralModel.name}: ${neuralModel.w}x${neuralModel.h} -> ${neuralModel.scale}x int8 planar`;
      }
    }
    this.lastMode = mode;

    // Dispatch shader path matching GlPipeline.kt drawFrameUnsafe()
    if (!settings.bypass && mode >= MODE_GPU) {
      if (settings.upscaler === UPSCALER_GSR && this.gsrProg && this.gsrCopyProg) {
        this.drawGsr(activeInputTex, capW, capH, outW, outH, settings);
      } else {
        this.drawCatmull(activeInputTex, capW, capH, outW, outH, settings);
      }
    } else {
      this.drawLook(activeInputTex, capW, capH, outW, outH, settings);
    }

    this.dgDrawn++;
    this.dgDrawMsAcc += performance.now() - tStart;
    this.publishDiag();
  }

  orbitDlss5Camera(dx: number, dy: number): void {
    if (!this.dlss5Engine) return;
    this.dlss5Engine.yawOffset += dx * 0.008;
    this.dlss5Engine.pitchOffset = Math.max(-1.2, Math.min(2.5, this.dlss5Engine.pitchOffset + dy * 0.008));
  }

  drawDlss5Frame(timeSec: number, settings: LookState): void {
    if (this.released || !this.dlss5Engine) return;
    const tStart = performance.now();
    this.dgCaptured++;

    const outW = this.canvas.width;
    const outH = this.canvas.height;
    const useDlss5 = !settings.bypass && settings.mode === MODE_DLSS5;

    this.dlss5Engine.renderDlss5Frame(outW, outH, timeSec, {
      captureHalf: settings.captureHalf,
      bypass: settings.bypass,
      upSharp: settings.upSharp,
      microRelief: settings.microRelief,
      dehaze: settings.dehaze,
      shadowLift: settings.shadowLift,
      hdr: settings.hdr,
      filmic: settings.filmic,
      glow: settings.glow,
      exposure: settings.exposure,
      contrast: settings.contrast,
      saturation: settings.saturation,
      grade: settings.grade,
      neuralLighting: useDlss5 ? settings.neuralLighting : 0,
      frameGen: useDlss5 && settings.frameGen,
      lutTex: this.lutTex,
      lutSize: this.lutSize,
      lutAmt: settings.lut,
    });

    const dt = this.dlss5Engine.telemetry;
    this.telemetry.captureWidth = dt.renderW;
    this.telemetry.captureHeight = dt.renderH;
    this.telemetry.outputWidth = dt.outputW;
    this.telemetry.outputHeight = dt.outputH;
    this.telemetry.dlss5Tier = dt.tier;
    this.telemetry.dlss5GbufMs = dt.gbufMs;
    this.telemetry.dlss5ReconMs = dt.reconMs;
    this.telemetry.dlss5FrameGen = dt.frameGenActive;

    const tierLabel =
      dt.tier === 1
        ? 'Tier 1: 3D-Guided Neural + TAAU'
        : dt.tier === 2
        ? 'Tier 2: SGSR2 TAAU (budget throttle)'
        : 'Tier 3: SGSR1 Spatial fallback';

    if (settings.bypass) {
      this.telemetry.status = '3D Engine Bypass (raw half-res G-buffer)';
      this.telemetry.stats = `Render ${dt.renderW}x${dt.renderH} -> ${dt.outputW}x${dt.outputH}`;
    } else {
      this.telemetry.status = `DLSS-5 ${tierLabel} · G-Buf ${dt.gbufMs.toFixed(1)} ms · Recon ${dt.reconMs.toFixed(1)} ms`;
      this.telemetry.stats = `${dt.frameGenActive ? 'FG 2x (120Hz cadence)' : 'FG off'} · ${dt.hasFloatGbuf ? 'RGBA16F MV' : 'RGBA8 MV'} · 16-phase Halton TAAU`;
    }

    this.lastMode = settings.mode;
    this.dgDrawn++;
    this.dgDrawMsAcc += performance.now() - tStart;
    this.publishDiag();
  }

  private drawGsr(
    inputTex: WebGLTexture,
    capW: number,
    capH: number,
    outW: number,
    outH: number,
    s: LookState,
  ): void {
    const gl = this.gl;
    this.ensureGsrTarget(capW, capH);

    // Pass 1: FXAA3 edge AA + top-down copy into gsrTex
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gsrFbo);
    gl.viewport(0, 0, capW, capH);
    gl.useProgram(this.gsrCopyProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(this.uGsrCopy.uTex, 0);
    gl.uniform2f(this.uGsrCopy.uTexel, 1 / capW, 1 / capH);
    gl.uniform1f(this.uGsrCopy.uAa, s.aa);
    this.drawQuad();

    // Pass 2: Qualcomm SGSR1 + vibrance + clarity + HyperReal 3D LUT
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.gsrProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.gsrTex);
    gl.uniform1i(this.uGsr.ps0, 0);
    gl.uniform4f(this.uGsr.ViewportInfo, 1 / capW, 1 / capH, capW, capH);
    gl.uniform1f(this.uGsr.uEdgeSharpness, 1 + s.upSharp);
    gl.uniform1f(this.uGsr.uClarity, s.upClarity);
    gl.uniform1f(this.uGsr.uVibrance, s.upVibrance);
    gl.uniform1f(this.uGsr.uMicroRelief, s.microRelief);
    gl.uniform1f(this.uGsr.uDehaze, s.dehaze);
    gl.uniform1f(this.uGsr.uShadowLift, s.shadowLift);
    gl.uniform1f(this.uGsr.uHdr, s.hdr);
    gl.uniform1f(this.uGsr.uFilmic, s.filmic);
    gl.uniform1f(this.uGsr.uGlow, s.glow);
    gl.uniform1f(this.uGsr.uExposure, s.exposure);
    gl.uniform1f(this.uGsr.uContrast, s.contrast);
    gl.uniform1f(this.uGsr.uSat, s.saturation);
    gl.uniform1f(this.uGsr.uGrade, s.grade);
    gl.uniform1f(this.uGsr.uBlack, s.blackPt);
    gl.uniform1f(this.uGsr.uVig, s.vignette);
    this.bindLut(this.uGsr.uLutAmt, this.uGsr.uLutMap, this.uGsr.uLut, s.lut);
    this.drawQuad();
  }

  private drawCatmull(
    inputTex: WebGLTexture,
    capW: number,
    capH: number,
    outW: number,
    outH: number,
    s: LookState,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.upProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(this.uUp.uTex, 0);
    gl.uniform2f(this.uUp.uTexel, 1 / capW, 1 / capH);
    gl.uniform1f(this.uUp.uSharp, s.upSharp);
    gl.uniform1f(this.uUp.uMicroRelief, s.microRelief);
    gl.uniform1f(this.uUp.uDehaze, s.dehaze);
    gl.uniform1f(this.uUp.uShadowLift, s.shadowLift);
    gl.uniform1f(this.uUp.uHdr, s.hdr);
    gl.uniform1f(this.uUp.uFilmic, s.filmic);
    gl.uniform1f(this.uUp.uExposure, s.exposure);
    gl.uniform1f(this.uUp.uContrast, s.contrast);
    gl.uniform1f(this.uUp.uSat, s.saturation);
    gl.uniform1f(this.uUp.uGrade, s.grade);
    gl.uniform1f(this.uUp.uBlack, s.blackPt);
    this.bindLut(this.uUp.uLutAmt, this.uUp.uLutMap, this.uUp.uLut, s.lut);
    this.drawQuad();
  }

  private drawLook(
    inputTex: WebGLTexture,
    capW: number,
    capH: number,
    outW: number,
    outH: number,
    s: LookState,
  ): void {
    const gl = this.gl;
    const useMotion = !s.bypass && s.motion > 0.01;
    if (useMotion) {
      this.ensureMotionFbo(outW, outH);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.fboTex[this.cur],
        0,
      );
    } else {
      this.hasPrev = false;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.lookProg);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTex[1 - this.cur]);
    gl.uniform1i(this.uLook.uPrev, 1);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(this.uLook.uTex, 0);

    this.bindLut(this.uLook.uLutAmt, this.uLook.uLutMap, this.uLook.uLut, s.lut);

    gl.uniform1f(this.uLook.uBypass, s.bypass ? 1 : 0);
    gl.uniform2f(this.uLook.uRes, outW, outH);
    gl.uniform1f(this.uLook.uHasPrev, useMotion && this.hasPrev ? 1 : 0);
    gl.uniform1f(this.uLook.uMotion, useMotion ? s.motion : 0);
    gl.uniform1f(this.uLook.uVig, s.vignette);
    gl.uniform1f(this.uLook.uChroma, s.chroma);
    gl.uniform1f(this.uLook.uHdr, s.hdr);
    gl.uniform1f(this.uLook.uBlack, s.blackPt);
    gl.uniform1f(this.uLook.uMicroRelief, s.microRelief);
    gl.uniform1f(this.uLook.uDehaze, s.dehaze);
    gl.uniform1f(this.uLook.uShadowLift, s.shadowLift);
    gl.uniform2f(this.uLook.uTexel, 1 / capW, 1 / capH);
    gl.uniform1f(this.uLook.uSharp, s.sharpen);
    gl.uniform1f(this.uLook.uClarity, s.clarity);
    gl.uniform1f(this.uLook.uContrast, s.contrast);
    gl.uniform1f(this.uLook.uSat, s.saturation);
    gl.uniform1f(this.uLook.uFilmic, s.filmic);
    gl.uniform1f(this.uLook.uExposure, s.exposure);
    gl.uniform1f(this.uLook.uGrade, s.grade);
    gl.uniform1f(this.uLook.uVibrance, s.vibrance);
    gl.uniform1f(this.uLook.uGlow, s.glow);

    this.drawQuad();

    if (useMotion) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, outW, outH);
      gl.useProgram(this.blitProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.fboTex[this.cur]);
      gl.uniform1i(this.uBlit.uTex2, 0);
      gl.uniform2f(this.uBlit.uRes, outW, outH);
      this.drawQuad();
      this.cur = 1 - this.cur;
      this.hasPrev = true;
    }
  }

  private publishDiag(): void {
    const now = performance.now();
    const dt = (now - this.dgSince) / 1000;
    if (dt < 1.0) return;
    const capFps = this.dgCaptured / dt;
    const drawFps = this.dgDrawn / dt;
    const msPerFrame = this.dgDrawn > 0 ? this.dgDrawMsAcc / this.dgDrawn : 0;
    this.telemetry.diag = `capture ${capFps.toFixed(0)} frames/s, drawn ${drawFps.toFixed(0)}/s, swap ${msPerFrame.toFixed(1)} ms/frame`;
    this.dgCaptured = 0;
    this.dgDrawn = 0;
    this.dgDrawMsAcc = 0;
    this.dgSince = now;
  }

  release(): void {
    this.released = true;
    this.dlss5Engine?.release();
    this.dlss5Engine = null;
    const gl = this.gl;
    if (this.srcTex) gl.deleteTexture(this.srcTex);
    if (this.halfTex) gl.deleteTexture(this.halfTex);
    if (this.halfFbo) gl.deleteFramebuffer(this.halfFbo);
    if (this.lutTex) gl.deleteTexture(this.lutTex);
    if (this.gsrTex) gl.deleteTexture(this.gsrTex);
    if (this.gsrFbo) gl.deleteFramebuffer(this.gsrFbo);
    if (this.fboTex[0]) gl.deleteTexture(this.fboTex[0]);
    if (this.fboTex[1]) gl.deleteTexture(this.fboTex[1]);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.lookProg) gl.deleteProgram(this.lookProg);
    if (this.upProg) gl.deleteProgram(this.upProg);
    if (this.gsrCopyProg) gl.deleteProgram(this.gsrCopyProg);
    if (this.gsrProg) gl.deleteProgram(this.gsrProg);
    if (this.blitProg) gl.deleteProgram(this.blitProg);
    if (this.quadBuf) gl.deleteBuffer(this.quadBuf);
    if (this.quadVao) gl.deleteVertexArray(this.quadVao);
  }
}
