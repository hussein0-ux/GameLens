import {
  applySubpixelJitter,
  DlssTier,
  FrameBudgetGovernor,
  getHaltonJitter,
  Mat4,
  mat4LookAt,
  mat4Multiply,
  mat4Perspective,
  mat4Transform,
} from '../core/dlss5Math';
import {
  DLSS5_FRAMEGEN_FS,
  DLSS5_NEURAL_TAAU_FS,
  DLSS5_PRESENT_FS,
  GBUFFER_FRAGMENT_SRC,
  GBUFFER_VERTEX_SRC,
} from './dlss5Shaders';
import { VERTEX_SRC } from './shaders';

interface MeshGroup {
  vao: WebGLVertexArrayObject;
  vbo: WebGLBuffer;
  ibo: WebGLBuffer;
  indexCount: number;
}

export interface Dlss5Telemetry {
  tier: DlssTier;
  gbufMs: number;
  reconMs: number;
  renderW: number;
  renderH: number;
  outputW: number;
  outputH: number;
  frameGenActive: boolean;
  hasFloatGbuf: boolean;
}

function createBoxMesh(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  d: number,
  r: number,
  g: number,
  b: number,
  roughness: number,
): MeshGroup {
  const hw = w / 2;
  const hh = h / 2;
  const hd = d / 2;
  const faces: [number[], number[], number[][]][] = [
    [[0, 0, 1], [0, 0], [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]]],
    [[0, 0, -1], [0, 0], [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]]],
    [[0, 1, 0], [0, 0], [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]]],
    [[0, -1, 0], [0, 0], [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]]],
    [[1, 0, 0], [0, 0], [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]]],
    [[-1, 0, 0], [0, 0], [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]]],
  ];
  const verts: number[] = [];
  const indices: number[] = [];
  const uvs = [[0, 0], [1, 0], [1, 1], [0, 1]];

  faces.forEach(([norm, _unused, corners], fi) => {
    const base = fi * 4;
    corners.forEach((pos, ci) => {
      verts.push(
        pos[0], pos[1], pos[2],
        norm[0], norm[1], norm[2],
        uvs[ci][0], uvs[ci][1],
        r, g, b, roughness,
      );
    });
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  });

  const vao = gl.createVertexArray()!;
  const vbo = gl.createBuffer()!;
  const ibo = gl.createBuffer()!;
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

  const stride = 12 * 4;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * 4);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 6 * 4);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 8 * 4);
  gl.bindVertexArray(null);

  return { vao, vbo, ibo, indexCount: indices.length };
}

export class Dlss5Engine {
  private gbufProg: WebGLProgram;
  private taauProg: WebGLProgram;
  private frameGenProg: WebGLProgram;
  private presentProg: WebGLProgram;

  private uGbuf: Record<string, WebGLUniformLocation | null> = {};
  private uTaau: Record<string, WebGLUniformLocation | null> = {};
  private uFg: Record<string, WebGLUniformLocation | null> = {};
  private uPresent: Record<string, WebGLUniformLocation | null> = {};

  private floorMesh: MeshGroup;
  private pillarMesh: MeshGroup;
  private crateMesh: MeshGroup;
  private droneMesh: MeshGroup;

  private gbufFbo: WebGLFramebuffer | null = null;
  private gColorTex: WebGLTexture | null = null;
  private gMotionDepthTex: WebGLTexture | null = null;
  private gNormalMatTex: WebGLTexture | null = null;
  private gDepthRb: WebGLRenderbuffer | null = null;
  private renderW = 0;
  private renderH = 0;

  private histFbo: WebGLFramebuffer | null = null;
  private histTex: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  private fgFbo: WebGLFramebuffer | null = null;
  private fgTex: WebGLTexture | null = null;
  private outW = 0;
  private outH = 0;
  private curHist = 0;
  private hasHistory = false;
  private renderedTaauCount = 0;

  private hasFloatGbuf = false;
  private frameIndex = 0;
  private prevViewProj: Mat4 | null = null;
  private prevDrone1Mvp: Mat4 | null = null;
  private prevDrone2Mvp: Mat4 | null = null;

  public yawOffset = 0;
  public pitchOffset = 0;
  public governor = new FrameBudgetGovernor();
  public telemetry: Dlss5Telemetry = {
    tier: 1,
    gbufMs: 0,
    reconMs: 0,
    renderW: 0,
    renderH: 0,
    outputW: 0,
    outputH: 0,
    frameGenActive: false,
    hasFloatGbuf: false,
  };

  constructor(
    private readonly gl: WebGL2RenderingContext,
    buildProgram: (vs: string, fs: string) => WebGLProgram,
    private readonly drawQuad: () => void,
  ) {
    this.hasFloatGbuf = Boolean(
      gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'),
    );
    this.telemetry.hasFloatGbuf = this.hasFloatGbuf;

    this.gbufProg = buildProgram(GBUFFER_VERTEX_SRC, GBUFFER_FRAGMENT_SRC);
    for (const k of ['uModel', 'uMvp', 'uPrevMvp', 'uTime', 'uObjId', 'uFloatGbuf', 'uJitter', 'uRenderRes']) {
      this.uGbuf[k] = gl.getUniformLocation(this.gbufProg, k);
    }

    this.taauProg = buildProgram(VERTEX_SRC, DLSS5_NEURAL_TAAU_FS);
    for (const k of [
      'uGColor', 'uGMotionDepth', 'uGNormalMat', 'uHistoryPrev',
      'uRenderRes', 'uOutRes', 'uJitter', 'uFloatGbuf', 'uHasHistory', 'uNeuralLight', 'uTier',
    ]) {
      this.uTaau[k] = gl.getUniformLocation(this.taauProg, k);
    }

    this.frameGenProg = buildProgram(VERTEX_SRC, DLSS5_FRAMEGEN_FS);
    for (const k of ['uHistoryPrev', 'uHistoryCurr', 'uGMotionDepth', 'uFloatGbuf', 'uBlendT']) {
      this.uFg[k] = gl.getUniformLocation(this.frameGenProg, k);
    }

    this.presentProg = buildProgram(VERTEX_SRC, DLSS5_PRESENT_FS);
    for (const k of [
      'uTex', 'uOutRes', 'uSharp', 'uMicroRelief', 'uDehaze', 'uShadowLift',
      'uHdr', 'uFilmic', 'uGlow', 'uExposure', 'uContrast', 'uSat', 'uGrade',
      'uBypass', 'uShowHud', 'uLut', 'uLutAmt', 'uLutMap',
    ]) {
      this.uPresent[k] = gl.getUniformLocation(this.presentProg, k);
    }

    // Build procedural tactical courtyard scene geometry
    this.floorMesh = createBoxMesh(gl, 40, 0.4, 40, 0.56, 0.50, 0.42, 0.72);
    this.pillarMesh = createBoxMesh(gl, 1.3, 6.0, 1.3, 0.78, 0.67, 0.52, 0.85);
    this.crateMesh = createBoxMesh(gl, 1.5, 1.2, 1.5, 0.34, 0.40, 0.32, 0.55);
    this.droneMesh = createBoxMesh(gl, 1.4, 0.55, 0.9, 0.24, 0.29, 0.36, 0.18);
  }

  private ensureTargets(rW: number, rH: number, oW: number, oH: number): void {
    const gl = this.gl;
    if (this.renderW !== rW || this.renderH !== rH || !this.gbufFbo) {
      if (this.gColorTex) gl.deleteTexture(this.gColorTex);
      if (this.gMotionDepthTex) gl.deleteTexture(this.gMotionDepthTex);
      if (this.gNormalMatTex) gl.deleteTexture(this.gNormalMatTex);
      if (this.gDepthRb) gl.deleteRenderbuffer(this.gDepthRb);
      if (this.gbufFbo) gl.deleteFramebuffer(this.gbufFbo);

      this.renderW = rW;
      this.renderH = rH;

      const makeTex = (internalFmt: number, fmt: number, type: number, filter: number) => {
        const t = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, rW, rH, 0, fmt, type, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
      };

      this.gColorTex = makeTex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
      this.gMotionDepthTex = this.hasFloatGbuf
        ? makeTex(gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST)
        : makeTex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);
      this.gNormalMatTex = makeTex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);

      this.gDepthRb = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.gDepthRb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, rW, rH);

      this.gbufFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.gbufFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.gColorTex, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gMotionDepthTex, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, this.gNormalMatTex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.gDepthRb);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);

      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE && this.hasFloatGbuf) {
        this.hasFloatGbuf = false;
        this.telemetry.hasFloatGbuf = false;
        gl.deleteTexture(this.gMotionDepthTex);
        this.gMotionDepthTex = makeTex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.gMotionDepthTex, 0);
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.hasHistory = false;
      this.renderedTaauCount = 0;
    }

    if (this.outW !== oW || this.outH !== oH || !this.histFbo) {
      if (this.histTex[0]) gl.deleteTexture(this.histTex[0]);
      if (this.histTex[1]) gl.deleteTexture(this.histTex[1]);
      if (this.fgTex) gl.deleteTexture(this.fgTex);
      if (this.histFbo) gl.deleteFramebuffer(this.histFbo);
      if (this.fgFbo) gl.deleteFramebuffer(this.fgFbo);

      this.outW = oW;
      this.outH = oH;
      this.histFbo = gl.createFramebuffer();
      for (let i = 0; i < 2; i++) {
        const t = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, oW, oH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.histTex[i] = t;
      }

      this.fgTex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this.fgTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, oW, oH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.fgFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fgFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fgTex, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.hasHistory = false;
      this.renderedTaauCount = 0;
    }
  }

  private drawMesh(mesh: MeshGroup, model: Mat4, vp: Mat4, prevVp: Mat4, objId: number): void {
    const gl = this.gl;
    const mvp = mat4Multiply(vp, model);
    const prevMvp = mat4Multiply(prevVp, model);
    gl.uniformMatrix4fv(this.uGbuf.uModel, false, model);
    gl.uniformMatrix4fv(this.uGbuf.uMvp, false, mvp);
    gl.uniformMatrix4fv(this.uGbuf.uPrevMvp, false, prevMvp);
    gl.uniform1f(this.uGbuf.uObjId, objId);
    gl.bindVertexArray(mesh.vao);
    gl.drawElements(gl.TRIANGLES, mesh.indexCount, gl.UNSIGNED_SHORT, 0);
  }

  renderDlss5Frame(
    outW: number,
    outH: number,
    timeSec: number,
    options: {
      captureHalf: boolean;
      bypass: boolean;
      upSharp: number;
      microRelief: number;
      dehaze: number;
      shadowLift: number;
      hdr: number;
      filmic: number;
      glow: number;
      exposure: number;
      contrast: number;
      saturation: number;
      grade: number;
      neuralLighting: number;
      frameGen: boolean;
      lutTex: WebGLTexture | null;
      lutSize: number;
      lutAmt: number;
    },
  ): void {
    const gl = this.gl;
    const scaleDiv = options.captureHalf ? 2 : 1;
    const rW = Math.max(320, Math.floor(outW / scaleDiv));
    const rH = Math.max(180, Math.floor(outH / scaleDiv));
    this.ensureTargets(rW, rH, outW, outH);

    const frameStart = performance.now();
    const isFrameGenMidpoint =
      !options.bypass &&
      options.frameGen &&
      this.hasHistory &&
      this.renderedTaauCount >= 2 &&
      this.telemetry.tier === 1 &&
      (this.frameIndex & 1) === 1;

    if (isFrameGenMidpoint) {
      // Frame Generation intermediate pass: synthesize frame N+0.5 from history + motion vectors without re-rendering G-buffer!
      const prevIdx = 1 - this.curHist;
      const currIdx = this.curHist;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fgFbo);
      gl.viewport(0, 0, outW, outH);
      gl.useProgram(this.frameGenProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.histTex[prevIdx]);
      gl.uniform1i(this.uFg.uHistoryPrev, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.histTex[currIdx]);
      gl.uniform1i(this.uFg.uHistoryCurr, 1);
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.gMotionDepthTex);
      gl.uniform1i(this.uFg.uGMotionDepth, 2);
      gl.uniform1f(this.uFg.uFloatGbuf, this.hasFloatGbuf ? 1 : 0);
      gl.uniform1f(this.uFg.uBlendT, 0.5);
      this.drawQuad();

      this.present(this.fgTex!, outW, outH, options);
      this.telemetry.reconMs = performance.now() - frameStart;
      this.telemetry.frameGenActive = true;
      this.frameIndex++;
      return;
    }

    // 1) Half-res jittered 3D G-buffer pass
    const [jx, jy] = options.bypass ? [0, 0] : getHaltonJitter(this.frameIndex);
    const orbit = timeSec * 0.35 + this.yawOffset;
    const eye: [number, number, number] = [
      Math.sin(orbit) * 11.5,
      2.9 + Math.max(-1.2, Math.min(2.5, this.pitchOffset)),
      Math.cos(orbit) * 11.5,
    ];
    const view = mat4LookAt(eye, [0, 1.3, 0], [0, 1, 0]);
    const baseProj = mat4Perspective((52 * Math.PI) / 180, outW / outH, 0.2, 80);
    const jitterProj = applySubpixelJitter(baseProj, jx, jy, rW, rH);
    const unjitteredVp = mat4Multiply(baseProj, view);
    const jitteredVp = mat4Multiply(jitterProj, view);
    const prevVp = this.prevViewProj ?? unjitteredVp;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gbufFbo);
    gl.viewport(0, 0, rW, rH);
    gl.enable(gl.DEPTH_TEST);
    gl.clearBufferfv(gl.COLOR, 0, [0.58, 0.70, 0.84, 1.0]);
    const mvZero = this.hasFloatGbuf ? 0.0 : 0.5;
    gl.clearBufferfv(gl.COLOR, 1, [mvZero, mvZero, 1.0, 0.0]);
    gl.clearBufferfv(gl.COLOR, 2, [0.5, 1.0, 0.5, 1.0]);
    gl.clearBufferfi(gl.DEPTH_STENCIL, 0, 1.0, 0);

    gl.useProgram(this.gbufProg);
    gl.uniform1f(this.uGbuf.uTime, timeSec);
    gl.uniform1f(this.uGbuf.uFloatGbuf, this.hasFloatGbuf ? 1 : 0);
    gl.uniform2f(this.uGbuf.uJitter, jx, jy);
    gl.uniform2f(this.uGbuf.uRenderRes, rW, rH);

    // Static courtyard geometry
    this.drawMesh(this.floorMesh, mat4Transform(0, -0.2, 0, 0), jitteredVp, prevVp, 0.1);
    const pillarCoords: [number, number][] = [[-4.5, -4.5], [4.5, -4.5], [-4.5, 4.5], [4.5, 4.5]];
    for (const [px, pz] of pillarCoords) {
      this.drawMesh(this.pillarMesh, mat4Transform(px, 3.0, pz, 0), jitteredVp, prevVp, 0.3);
    }
    this.drawMesh(this.crateMesh, mat4Transform(-1.8, 0.6, 1.4, 0.4), jitteredVp, prevVp, 0.5);
    this.drawMesh(this.crateMesh, mat4Transform(2.1, 0.6, -1.6, -0.3), jitteredVp, prevVp, 0.5);

    // Animated hovering drones with exact object + camera PrevMVP motion vectors
    const d1Model = mat4Transform(Math.cos(timeSec * 1.4) * 3.2, 1.8 + Math.sin(timeSec * 2.5) * 0.35, Math.sin(timeSec * 1.4) * 3.2, -timeSec * 1.4);
    const d1Mvp = mat4Multiply(jitteredVp, d1Model);
    const d1UnjMvp = mat4Multiply(unjitteredVp, d1Model);
    gl.uniformMatrix4fv(this.uGbuf.uModel, false, d1Model);
    gl.uniformMatrix4fv(this.uGbuf.uMvp, false, d1Mvp);
    gl.uniformMatrix4fv(this.uGbuf.uPrevMvp, false, this.prevDrone1Mvp ?? d1UnjMvp);
    gl.uniform1f(this.uGbuf.uObjId, 0.9);
    gl.bindVertexArray(this.droneMesh.vao);
    gl.drawElements(gl.TRIANGLES, this.droneMesh.indexCount, gl.UNSIGNED_SHORT, 0);
    this.prevDrone1Mvp = d1UnjMvp;

    const d2Model = mat4Transform(Math.sin(timeSec * 1.1) * 2.4, 2.7, -2.5 + Math.cos(timeSec * 1.6) * 2.0, timeSec * 0.9);
    const d2Mvp = mat4Multiply(jitteredVp, d2Model);
    const d2UnjMvp = mat4Multiply(unjitteredVp, d2Model);
    gl.uniformMatrix4fv(this.uGbuf.uModel, false, d2Model);
    gl.uniformMatrix4fv(this.uGbuf.uMvp, false, d2Mvp);
    gl.uniformMatrix4fv(this.uGbuf.uPrevMvp, false, this.prevDrone2Mvp ?? d2UnjMvp);
    gl.uniform1f(this.uGbuf.uObjId, 0.9);
    gl.drawElements(gl.TRIANGLES, this.droneMesh.indexCount, gl.UNSIGNED_SHORT, 0);
    this.prevDrone2Mvp = d2UnjMvp;

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    this.prevViewProj = unjitteredVp;

    const gbufDone = performance.now();
    this.telemetry.gbufMs = gbufDone - frameStart;

    if (options.bypass) {
      // Bypass shows raw half-res aliased G-buffer directly
      this.present(this.gColorTex!, outW, outH, options);
      this.telemetry.reconMs = 0;
      this.telemetry.renderW = rW;
      this.telemetry.renderH = rH;
      this.telemetry.outputW = outW;
      this.telemetry.outputH = outH;
      this.telemetry.frameGenActive = false;
      this.frameIndex++;
      return;
    }

    // 2) 3D-Guided Neural Lighting + Temporal Super Resolution (TAAU)
    const nextHist = 1 - this.curHist;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.histFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.histTex[nextHist], 0);
    gl.viewport(0, 0, outW, outH);

    gl.useProgram(this.taauProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.gColorTex);
    gl.uniform1i(this.uTaau.uGColor, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.gMotionDepthTex);
    gl.uniform1i(this.uTaau.uGMotionDepth, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.gNormalMatTex);
    gl.uniform1i(this.uTaau.uGNormalMat, 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.histTex[this.curHist]);
    gl.uniform1i(this.uTaau.uHistoryPrev, 3);

    gl.uniform2f(this.uTaau.uRenderRes, rW, rH);
    gl.uniform2f(this.uTaau.uOutRes, outW, outH);
    gl.uniform2f(this.uTaau.uJitter, jx, jy);
    gl.uniform1f(this.uTaau.uFloatGbuf, this.hasFloatGbuf ? 1 : 0);
    gl.uniform1f(this.uTaau.uHasHistory, this.hasHistory ? 1 : 0);
    gl.uniform1f(this.uTaau.uNeuralLight, options.neuralLighting);
    gl.uniform1i(this.uTaau.uTier, this.telemetry.tier);
    this.drawQuad();

    this.curHist = nextHist;
    this.hasHistory = true;
    this.renderedTaauCount++;

    // 3) Present with CAS + native HUD + HyperReal 3D LUT
    this.present(this.histTex[this.curHist]!, outW, outH, options);

    const totalMs = performance.now() - frameStart;
    this.telemetry.reconMs = performance.now() - gbufDone;
    this.telemetry.tier = this.governor.onFrameTime(totalMs);
    this.telemetry.renderW = rW;
    this.telemetry.renderH = rH;
    this.telemetry.outputW = outW;
    this.telemetry.outputH = outH;
    this.telemetry.frameGenActive = options.frameGen && this.telemetry.tier === 1;
    this.frameIndex++;
  }

  private present(
    tex: WebGLTexture,
    outW: number,
    outH: number,
    options: {
      bypass: boolean;
      upSharp: number;
      microRelief: number;
      dehaze: number;
      shadowLift: number;
      hdr: number;
      filmic: number;
      glow: number;
      exposure: number;
      contrast: number;
      saturation: number;
      grade: number;
      lutTex: WebGLTexture | null;
      lutSize: number;
      lutAmt: number;
    },
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(this.presentProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.uPresent.uTex, 0);
    gl.uniform2f(this.uPresent.uOutRes, outW, outH);
    gl.uniform1f(this.uPresent.uSharp, options.upSharp);
    gl.uniform1f(this.uPresent.uMicroRelief, options.microRelief);
    gl.uniform1f(this.uPresent.uDehaze, options.dehaze);
    gl.uniform1f(this.uPresent.uShadowLift, options.shadowLift);
    gl.uniform1f(this.uPresent.uHdr, options.hdr);
    gl.uniform1f(this.uPresent.uFilmic, options.filmic);
    gl.uniform1f(this.uPresent.uGlow, options.glow);
    gl.uniform1f(this.uPresent.uExposure, options.exposure);
    gl.uniform1f(this.uPresent.uContrast, options.contrast);
    gl.uniform1f(this.uPresent.uSat, options.saturation);
    gl.uniform1f(this.uPresent.uGrade, options.grade);
    gl.uniform1f(this.uPresent.uBypass, options.bypass ? 1 : 0);
    gl.uniform1f(this.uPresent.uShowHud, options.bypass ? 0 : 1);

    gl.uniform1i(this.uPresent.uLut, 2);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, options.lutTex);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(this.uPresent.uLutAmt, options.lutTex ? options.lutAmt : 0);
    if (options.lutSize > 0) {
      gl.uniform2f(this.uPresent.uLutMap, (options.lutSize - 1) / options.lutSize, 0.5 / options.lutSize);
    }
    this.drawQuad();
  }

  release(): void {
    const gl = this.gl;
    for (const m of [this.floorMesh, this.pillarMesh, this.crateMesh, this.droneMesh]) {
      gl.deleteVertexArray(m.vao);
      gl.deleteBuffer(m.vbo);
      gl.deleteBuffer(m.ibo);
    }
    if (this.gColorTex) gl.deleteTexture(this.gColorTex);
    if (this.gMotionDepthTex) gl.deleteTexture(this.gMotionDepthTex);
    if (this.gNormalMatTex) gl.deleteTexture(this.gNormalMatTex);
    if (this.gDepthRb) gl.deleteRenderbuffer(this.gDepthRb);
    if (this.gbufFbo) gl.deleteFramebuffer(this.gbufFbo);
    if (this.histTex[0]) gl.deleteTexture(this.histTex[0]);
    if (this.histTex[1]) gl.deleteTexture(this.histTex[1]);
    if (this.histFbo) gl.deleteFramebuffer(this.histFbo);
    if (this.fgTex) gl.deleteTexture(this.fgTex);
    if (this.fgFbo) gl.deleteFramebuffer(this.fgFbo);
    gl.deleteProgram(this.gbufProg);
    gl.deleteProgram(this.taauProg);
    gl.deleteProgram(this.frameGenProg);
    gl.deleteProgram(this.presentProg);
  }
}
