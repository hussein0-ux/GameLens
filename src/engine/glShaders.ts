export const VERTEX_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform float uFlipY;
out vec2 vUv;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
  vUv = vec2(aUv.x, uFlipY > 0.5 ? 1.0 - aUv.y : aUv.y);
}
`;

export const BLIT_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

export const BLIT_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform sampler2D uTex2;
uniform vec2 uRes;
out vec4 outColor;
void main() {
  outColor = vec4(texture(uTex2, gl_FragCoord.xy / uRes).rgb, 1.0);
}
`;

/**
 * Downsamples or copies the source frame into a 2D texture (used for half-res capture simulation
 * and split-screen raw comparison).
 */
export const COPY_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform sampler2D uTex;
in vec2 vUv;
out vec4 outColor;
void main() {
  outColor = vec4(texture(uTex, vUv).rgb, 1.0);
}
`;

/**
 * FXAA 3 Console edge anti-aliasing pass (from GlPipeline.kt GSR_COPY_FRAGMENT_SRC).
 * Draws upside-down into gsrTex when uFlipCopy is 1 so row 0 matches top-left SGSR coordinates.
 */
export const GSR_COPY_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
  gl_Position = vec4(aPos.x, -aPos.y, 0.0, 1.0);
  vUv = aUv;
}
`;

export const GSR_COPY_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uAa;
in vec2 vUv;
out vec4 outColor;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec3 m = texture(uTex, vUv).rgb;
  if (uAa <= 0.0) {
    outColor = vec4(m, 1.0);
    return;
  }
  float nw = luma(texture(uTex, vUv + vec2(-1.0, -1.0) * uTexel).rgb);
  float ne = luma(texture(uTex, vUv + vec2( 1.0, -1.0) * uTexel).rgb);
  float sw = luma(texture(uTex, vUv + vec2(-1.0,  1.0) * uTexel).rgb);
  float se = luma(texture(uTex, vUv + vec2( 1.0,  1.0) * uTexel).rgb);
  float lm = luma(m);
  float lumaMin = min(lm, min(min(nw, ne), min(sw, se)));
  float lumaMax = max(lm, max(max(nw, ne), max(sw, se)));
  if (lumaMax - lumaMin < max(0.0312, lumaMax * 0.125)) {
    outColor = vec4(m, 1.0);
    return;
  }
  vec2 dir = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
  float dirReduce = max((nw + ne + sw + se) * 0.25 * 0.125, 1.0 / 128.0);
  float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * uTexel;
  vec3 a = 0.5 * (
    texture(uTex, vUv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture(uTex, vUv + dir * (2.0 / 3.0 - 0.5)).rgb
  );
  vec3 b = a * 0.5 + 0.25 * (
    texture(uTex, vUv - dir * 0.5).rgb +
    texture(uTex, vUv + dir * 0.5).rgb
  );
  float lb = luma(b);
  vec3 aa = (lb < lumaMin || lb > lumaMax) ? a : b;
  outColor = vec4(mix(m, aa, uAa), 1.0);
}
`;

export const GSR_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out highp vec4 in_TEXCOORD0;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
  in_TEXCOORD0 = vec4(aUv.x, 1.0 - aUv.y, 0.0, 0.0);
}
`;

/**
 * Qualcomm Snapdragon Game Super Resolution v1 (sgsr1_mobile.frag, BSD-3-Clause)
 * ported to WebGL2 (#version 300 es) using texelFetch-based gatherG() equivalent to GLES 3.1 textureGather().
 */
export const SGSR1_FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp int;

#define OperationMode 1
#define EdgeThreshold (8.0 / 255.0)

uniform float uEdgeSharpness;
uniform highp vec4 ViewportInfo[1];
uniform highp sampler2D ps0;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;
uniform float uClarity;
uniform float uVibrance;

in highp vec4 in_TEXCOORD0;
out vec4 out_Target0;

float fastLanczos2(float x) {
  float wA = x - 4.0;
  float wB = x * wA - wA;
  wA *= wA;
  return wB * wA;
}

vec2 weightY(float dx, float dy, float c, float std) {
  float x = ((dx * dx) + (dy * dy)) * 0.55 + clamp(abs(c) * std, 0.0, 1.0);
  float w = fastLanczos2(x);
  return vec2(w, w * c);
}

// Exact GLES 3.1 textureGather(ps0, coord, 1) replacement for WebGL2 (#version 300 es).
vec4 gatherG(highp vec2 coord, highp vec2 texSize) {
  ivec2 maxIdx = ivec2(texSize) - ivec2(1);
  ivec2 base = ivec2(floor(coord * texSize - 0.5));
  ivec2 p00 = clamp(base, ivec2(0), maxIdx);
  ivec2 p10 = clamp(base + ivec2(1, 0), ivec2(0), maxIdx);
  ivec2 p11 = clamp(base + ivec2(1, 1), ivec2(0), maxIdx);
  ivec2 p01 = clamp(base + ivec2(0, 1), ivec2(0), maxIdx);
  return vec4(
    texelFetch(ps0, p01, 0).g,
    texelFetch(ps0, p11, 0).g,
    texelFetch(ps0, p10, 0).g,
    texelFetch(ps0, p00, 0).g
  );
}

void main() {
  const int mode = OperationMode;
  float edgeThreshold = EdgeThreshold;
  float edgeSharpness = uEdgeSharpness;

  vec4 color;
  color.xyz = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyz;
  color.w = 1.0;

  highp vec2 texSize = ViewportInfo[0].zw;
  highp vec2 imgCoord = ((in_TEXCOORD0.xy * texSize) + vec2(-0.5, 0.5));
  highp vec2 imgCoordPixel = floor(imgCoord);
  highp vec2 coord = (imgCoordPixel * ViewportInfo[0].xy);
  vec2 pl = (imgCoord + (-imgCoordPixel));
  vec4 left = gatherG(coord, texSize);

  float edgeVote = abs(left.z - left.y) + abs(color.g - left.y) + abs(color.g - left.z);
  if (edgeVote > edgeThreshold) {
    coord.x += ViewportInfo[0].x;

    vec4 right = gatherG(coord + vec2(ViewportInfo[0].x, 0.0), texSize);
    vec4 upDown;
    upDown.xy = gatherG(coord + vec2(0.0, -ViewportInfo[0].y), texSize).wz;
    upDown.zw = gatherG(coord + vec2(0.0,  ViewportInfo[0].y), texSize).yx;

    float mean = (left.y + left.z + right.x + right.w) * 0.25;
    left = left - vec4(mean);
    right = right - vec4(mean);
    upDown = upDown - vec4(mean);
    color.w = color.g - mean;

    float sum = ((((abs(left.x) + abs(left.y)) + abs(left.z)) + abs(left.w))
               + (((abs(right.x) + abs(right.y)) + abs(right.z)) + abs(right.w)))
               + (((abs(upDown.x) + abs(upDown.y)) + abs(upDown.z)) + abs(upDown.w));
    float std = 2.181818 / max(sum, 1e-5);

    vec2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, std);
    aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, std);
    aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, std);
    aWY += weightY(pl.x, pl.y - 2.0, upDown.w, std);
    aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, std);
    aWY += weightY(pl.x, pl.y - 1.0, left.y, std);
    aWY += weightY(pl.x, pl.y, left.z, std);
    aWY += weightY(pl.x + 1.0, pl.y, left.w, std);
    aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, std);
    aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, std);
    aWY += weightY(pl.x - 2.0, pl.y, right.z, std);
    aWY += weightY(pl.x - 1.0, pl.y, right.w, std);

    float finalY = aWY.y / max(aWY.x, 1e-5);

    float maxY = max(max(left.y, left.z), max(right.x, right.w));
    float minY = min(min(left.y, left.z), min(right.x, right.w));
    finalY = clamp(edgeSharpness * finalY, minY, maxY);

    float deltaY = finalY - color.w;
    deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

    color.x = clamp(color.x + deltaY, 0.0, 1.0);
    color.y = clamp(color.y + deltaY, 0.0, 1.0);
    color.z = clamp(color.z + deltaY, 0.0, 1.0);
  }

  if (uVibrance > 0.0) {
    float sat = max(color.x, max(color.y, color.z)) - min(color.x, min(color.y, color.z));
    float luma = dot(color.xyz, vec3(0.299, 0.587, 0.114));
    color.xyz = clamp(vec3(luma) + (color.xyz - vec3(luma)) * (1.0 + uVibrance * (1.0 - sat)), 0.0, 1.0);
  }

  if (uClarity > 0.0) {
    highp vec2 o = ViewportInfo[0].xy * 1.5;
    vec3 lw = vec3(0.299, 0.587, 0.114);
    float blurLuma = 0.25 * (
      dot(textureLod(ps0, in_TEXCOORD0.xy + vec2( o.x,  o.y), 0.0).xyz, lw) +
      dot(textureLod(ps0, in_TEXCOORD0.xy + vec2(-o.x,  o.y), 0.0).xyz, lw) +
      dot(textureLod(ps0, in_TEXCOORD0.xy + vec2( o.x, -o.y), 0.0).xyz, lw) +
      dot(textureLod(ps0, in_TEXCOORD0.xy + vec2(-o.x, -o.y), 0.0).xyz, lw)
    );
    float d = (dot(color.xyz, lw) - blurLuma) * uClarity;
    color.xyz = clamp(color.xyz + vec3(d), 0.0, 1.0);
  }

  if (uLutAmt > 0.0) {
    highp vec3 lutCoord = color.xyz * uLutMap.x + uLutMap.y;
    color.xyz = mix(color.xyz, texture(uLut, lutCoord).rgb, uLutAmt);
  }

  color.w = 1.0;
  out_Target0 = color;
}
`;

/**
 * Catmull-Rom 2x upscale (9 bilinear fetches), deringing clamp, adaptive CAS sharpen, HyperReal LUT.
 */
export const UPSCALE_FRAGMENT_SRC = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uSharp;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

in vec2 vUv;
out vec4 outColor;

vec3 tex(vec2 p) { return texture(uTex, p).rgb; }

void main() {
  vec2 size = 1.0 / uTexel;
  vec2 pos = vUv * size;
  vec2 p1 = floor(pos - 0.5) + 0.5;
  vec2 f = pos - p1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 p0 = (p1 - 1.0) * uTexel;
  vec2 p3 = (p1 + 2.0) * uTexel;
  vec2 p12 = (p1 + w2 / w12) * uTexel;

  vec3 c = tex(vec2(p0.x, p0.y)) * w0.x * w0.y
         + tex(vec2(p12.x, p0.y)) * w12.x * w0.y
         + tex(vec2(p3.x, p0.y)) * w3.x * w0.y
         + tex(vec2(p0.x, p12.y)) * w0.x * w12.y
         + tex(vec2(p12.x, p12.y)) * w12.x * w12.y
         + tex(vec2(p3.x, p12.y)) * w3.x * w12.y
         + tex(vec2(p0.x, p3.y)) * w0.x * w3.y
         + tex(vec2(p12.x, p3.y)) * w12.x * w3.y
         + tex(vec2(p3.x, p3.y)) * w3.x * w3.y;

  vec2 d = 0.5 * uTexel;
  vec3 n = tex(vUv + vec2(0.0, -d.y));
  vec3 s = tex(vUv + vec2(0.0,  d.y));
  vec3 w = tex(vUv + vec2(-d.x, 0.0));
  vec3 e = tex(vUv + vec2( d.x, 0.0));
  vec3 m = tex(vUv);
  vec3 mn = min(m, min(min(n, s), min(w, e)));
  vec3 mx = max(m, max(max(n, s), max(w, e)));
  c = clamp(c, mn, mx);

  vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
  vec3 avg = (n + s + w + e) * 0.25;
  vec3 col = clamp(c + (c - avg) * uSharp * amp * 2.0, 0.0, 1.0);

  if (uLutAmt > 0.0) {
    col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
  }
  outColor = vec4(col, 1.0);
}
`;

/**
 * Full Look Shader (ported directly from GlPipeline.kt FRAGMENT_SRC).
 */
export const LOOK_FRAGMENT_SRC = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uSharp;
uniform float uClarity;
uniform float uContrast;
uniform float uSat;
uniform float uFilmic;
uniform float uExposure;
uniform float uGrade;
uniform float uVibrance;
uniform float uGlow;
uniform sampler2D uPrev;
uniform vec2 uRes;
uniform float uHasPrev;
uniform float uMotion;
uniform float uVig;
uniform float uChroma;
uniform float uHdr;
uniform float uBlack;
uniform float uBypass;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

const vec2 DIRS[8] = vec2[8](
  vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0), vec2(-0.7071, 0.7071),
  vec2(-1.0, 0.0), vec2(-0.7071, -0.7071), vec2(0.0, -1.0), vec2(0.7071, -0.7071)
);

in vec2 vUv;
out vec4 outColor;

vec3 tap(vec2 off) { return texture(uTex, vUv + off * uTexel).rgb; }
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  if (uBypass > 0.5) {
    outColor = vec4(texture(uTex, vUv).rgb, 1.0);
    return;
  }
  vec2 suv = gl_FragCoord.xy / uRes;
  vec2 cd = (suv - 0.5) * uChroma * 8.0;
  vec3 e = vec3(tap(cd).r, tap(vec2(0.0)).g, tap(-cd).b);
  vec3 n = tap(vec2(0.0, -1.0));
  vec3 s = tap(vec2(0.0,  1.0));
  vec3 w = tap(vec2(-1.0, 0.0));
  vec3 ea = tap(vec2( 1.0, 0.0));

  vec3 mn = min(e, min(min(n, s), min(w, ea)));
  vec3 mx = max(e, max(max(n, s), max(w, ea)));
  vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
  float peak = mix(-0.125, -0.2, uSharp);
  vec3 wgt = amp * peak;
  vec3 sharp = (e + (n + s + w + ea) * wgt) / (1.0 + 4.0 * wgt);
  vec3 col = mix(e, sharp, step(0.001, uSharp));

  vec3 blur = (tap(vec2(-2.0, 0.0)) + tap(vec2(2.0, 0.0)) + tap(vec2(0.0, -2.0)) + tap(vec2(0.0, 2.0))) * 0.25;
  float midMask = 1.0 - abs(luma(col) * 2.0 - 1.0);
  col += (col - blur) * uClarity * (0.5 + 0.5 * midMask);
  col = clamp(col, 0.0, 1.0);

  vec3 glowAcc = vec3(0.0);
  for (int i = 0; i < 8; i++) {
    vec3 gs = tap(DIRS[i] * 9.0);
    glowAcc += max(gs - 0.55, 0.0);
  }
  col += (glowAcc / 8.0) * (uGlow * 1.5);
  float wideW = smoothstep(0.4, 1.0, uGlow);
  if (wideW > 0.0) {
    vec3 wideAcc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
      wideAcc += max(tap(DIRS[i] * 24.0) - 0.6, vec3(0.0));
    }
    col += (wideAcc / 8.0) * (wideW * 1.2);
  }

  if (uHdr > 0.001) {
    vec3 wide = (
      tap(vec2(-14.0, -14.0)) + tap(vec2(14.0, -14.0)) +
      tap(vec2(-14.0,  14.0)) + tap(vec2(14.0,  14.0)) +
      tap(vec2(-26.0,   0.0)) + tap(vec2(26.0,   0.0)) +
      tap(vec2(  0.0, -26.0)) + tap(vec2( 0.0,  26.0))
    ) * 0.125;
    float localL = luma(wide);
    float gain = clamp(pow(0.42 / (localL + 0.04), 0.5), 0.92, 1.7);
    col *= mix(1.0, gain, uHdr * 0.55);
    float lc = luma(col);
    float hi = smoothstep(0.62, 1.0, lc);
    col += col * hi * uHdr * 0.30;
    col = mix(col, col / (1.0 + max(col - 0.85, 0.0) * 1.2), uHdr);
  }

  col *= uExposure;
  vec3 curved = clamp(col * col * (3.0 - 2.0 * col), 0.0, 1.0);
  col = mix(clamp(col, 0.0, 1.0), curved, uFilmic);
  col = (col - 0.5) * uContrast + 0.5;

  float l0 = luma(col);
  float shadowsW = 1.0 - smoothstep(0.0, 0.55, l0);
  float highsW = smoothstep(0.45, 1.0, l0);
  col += uGrade * (vec3(-0.03, 0.03, 0.08) * shadowsW + vec3(0.08, 0.03, -0.05) * highsW);

  float chromaMax = max(col.r, max(col.g, col.b));
  float chromaMin = min(col.r, min(col.g, col.b));
  float satNow = chromaMax - chromaMin;
  col = mix(vec3(luma(col)), col, 1.0 + uVibrance * (1.0 - satNow));
  col = mix(vec3(luma(col)), col, uSat);

  float bp = uBlack * 0.07;
  col = max(col - bp, 0.0) / (1.0 - bp);

  col *= 1.0 - uVig * smoothstep(0.30, 0.80, length(suv - 0.5));
  col = clamp(col, 0.0, 1.0);
  if (uLutAmt > 0.0) {
    col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
  }
  if (uMotion > 0.0 && uHasPrev > 0.5) {
    col = mix(col, texture(uPrev, suv).rgb, uMotion);
  }
  outColor = vec4(col, 1.0);
}
`;

/**
 * Split-screen comparison shader: left of uSplitX shows raw input (or bypass),
 * right of uSplitX shows processed output.
 */
export const SPLIT_FRAGMENT_SRC = `#version 300 es
precision highp float;
uniform sampler2D uRawTex;
uniform sampler2D uOutTex;
uniform vec2 uRes;
uniform float uSplitX;
uniform float uSplitEnabled;
uniform float uZoom;
uniform vec2 uZoomCenter;
out vec4 outColor;

void main() {
  vec2 suv = gl_FragCoord.xy / uRes;
  vec2 sampleUv = uZoom > 1.01
    ? clamp((suv - 0.5) / uZoom + uZoomCenter, 0.0, 1.0)
    : suv;

  vec3 processed = texture(uOutTex, sampleUv).rgb;
  if (uSplitEnabled > 0.5 && suv.x < uSplitX) {
    vec3 rawCol = texture(uRawTex, vec2(sampleUv.x, 1.0 - sampleUv.y)).rgb;
    outColor = vec4(rawCol, 1.0);
  } else {
    outColor = vec4(processed, 1.0);
  }
}
`;
