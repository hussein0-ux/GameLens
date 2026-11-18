export const VERTEX_SRC = `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
out vec2 vUv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUv = aUv;
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

// Copy pass: draws the capture upside down on purpose so row 0 of the 2D texture is the top of the picture,
// matching GlPipeline.kt's GSR_COPY_VERTEX_SRC + GSR_COPY_FRAGMENT_SRC orientation.
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
uniform vec2 uTexel;   // one capture texel in uv units
uniform float uAa;     // 0 = plain copy
in vec2 vUv;
out vec4 outColor;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
// Edge anti-aliasing (FXAA 3 console variant). Only high-contrast pixels are touched, so texture detail is kept.
void main() {
    vec3 m = texture(uTex, vUv).rgb;
    if (uAa <= 0.0) { outColor = vec4(m, 1.0); return; }
    float nw = luma(texture(uTex, vUv + vec2(-1.0, -1.0) * uTexel).rgb);
    float ne = luma(texture(uTex, vUv + vec2( 1.0, -1.0) * uTexel).rgb);
    float sw = luma(texture(uTex, vUv + vec2(-1.0,  1.0) * uTexel).rgb);
    float se = luma(texture(uTex, vUv + vec2( 1.0,  1.0) * uTexel).rgb);
    float lm = luma(m);
    float lumaMin = min(lm, min(min(nw, ne), min(sw, se)));
    float lumaMax = max(lm, max(max(nw, ne), max(sw, se)));
    if (lumaMax - lumaMin < max(0.0312, lumaMax * 0.125)) { outColor = vec4(m, 1.0); return; }
    vec2 dir = vec2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));
    float dirReduce = max((nw + ne + sw + se) * 0.25 * 0.125, 1.0 / 128.0);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * uTexel;
    vec3 a = 0.5 * (texture(uTex, vUv + dir * (1.0 / 3.0 - 0.5)).rgb + texture(uTex, vUv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(uTex, vUv - dir * 0.5).rgb + texture(uTex, vUv + dir * 0.5).rgb);
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

// Faithful WebGL2 (#version 300 es) port of Qualcomm Snapdragon Game Super Resolution v1 (sgsr1_mobile.frag).
// Uses texelFetch 2x2 gather helper to match GLES 3.10 textureGather(ps0, coord, 1) identically.
export const GSR_FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp int;

#define EdgeThreshold (8.0 / 255.0)

uniform float uEdgeSharpness;
uniform highp vec4 ViewportInfo;
uniform mediump sampler2D ps0;

uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

uniform float uClarity;
uniform float uVibrance;
uniform float uMicroRelief;
uniform float uDehaze;
uniform float uShadowLift;
uniform float uHdr;
uniform float uFilmic;
uniform float uGlow;
uniform float uExposure;
uniform float uContrast;
uniform float uSat;
uniform float uGrade;
uniform float uBlack;
uniform float uVig;

in highp vec4 in_TEXCOORD0;
out vec4 out_Target0;

vec3 acesFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

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

// Exact GLES 3.10 textureGather(s, uv, 1) green-channel equivalent in WebGL2 GLSL ES 3.00.
vec4 gatherG(mediump sampler2D s, highp vec2 uv, highp vec2 texSize) {
    vec2 fc = uv * texSize - 0.5;
    ivec2 p0 = ivec2(floor(fc));
    ivec2 maxIdx = ivec2(texSize) - 1;
    ivec2 p00 = clamp(p0, ivec2(0), maxIdx);
    ivec2 p10 = clamp(p0 + ivec2(1, 0), ivec2(0), maxIdx);
    ivec2 p01 = clamp(p0 + ivec2(0, 1), ivec2(0), maxIdx);
    ivec2 p11 = clamp(p0 + ivec2(1, 1), ivec2(0), maxIdx);
    return vec4(
        texelFetch(s, p01, 0).g,
        texelFetch(s, p11, 0).g,
        texelFetch(s, p10, 0).g,
        texelFetch(s, p00, 0).g
    );
}

void main() {
    float edgeThreshold = EdgeThreshold;
    float edgeSharpness = uEdgeSharpness;

    vec4 color;
    color.xyz = textureLod(ps0, in_TEXCOORD0.xy, 0.0).xyz;

    highp vec2 imgCoord = ((in_TEXCOORD0.xy * ViewportInfo.zw) + vec2(-0.5, 0.5));
    highp vec2 imgCoordPixel = floor(imgCoord);
    highp vec2 coord = (imgCoordPixel * ViewportInfo.xy);
    vec2 pl = (imgCoord + (-imgCoordPixel));
    vec4 left = gatherG(ps0, coord, ViewportInfo.zw);

    float edgeVote = abs(left.z - left.y) + abs(color.y - left.y) + abs(color.y - left.z);
    if (edgeVote > edgeThreshold) {
        coord.x += ViewportInfo.x;

        vec4 right = gatherG(ps0, coord + vec2(ViewportInfo.x, 0.0), ViewportInfo.zw);
        vec4 upDown;
        upDown.xy = gatherG(ps0, coord + vec2(0.0, -ViewportInfo.y), ViewportInfo.zw).wz;
        upDown.zw = gatherG(ps0, coord + vec2(0.0, ViewportInfo.y), ViewportInfo.zw).yx;

        float mean = (left.y + left.z + right.x + right.w) * 0.25;
        left = left - vec4(mean);
        right = right - vec4(mean);
        upDown = upDown - vec4(mean);
        color.w = color.y - mean;

        float sum = (((((abs(left.x) + abs(left.y)) + abs(left.z)) + abs(left.w))
                    + (((abs(right.x) + abs(right.y)) + abs(right.z)) + abs(right.w)))
                    + (((abs(upDown.x) + abs(upDown.y)) + abs(upDown.z)) + abs(upDown.w)));
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

        color.x = clamp((color.x + deltaY), 0.0, 1.0);
        color.y = clamp((color.y + deltaY), 0.0, 1.0);
        color.z = clamp((color.z + deltaY), 0.0, 1.0);
    }

    if (uVibrance > 0.0) {
        float sat = max(color.x, max(color.y, color.z)) - min(color.x, min(color.y, color.z));
        float luma = dot(color.xyz, vec3(0.299, 0.587, 0.114));
        color.xyz = clamp(vec3(luma) + (color.xyz - vec3(luma)) * (1.0 + uVibrance * (1.0 - sat)), 0.0, 1.0);
    }

    if (uClarity > 0.0 || uMicroRelief > 0.0 || uGlow > 0.0 || uHdr > 0.0) {
        highp vec2 o1 = ViewportInfo.xy * 1.15;
        highp vec2 o2 = ViewportInfo.xy * 3.25;
        vec3 lw = vec3(0.299, 0.587, 0.114);

        vec3 s1 = textureLod(ps0, in_TEXCOORD0.xy + vec2( o1.x,  o1.y), 0.0).rgb;
        vec3 s2 = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o1.x,  o1.y), 0.0).rgb;
        vec3 s3 = textureLod(ps0, in_TEXCOORD0.xy + vec2( o1.x, -o1.y), 0.0).rgb;
        vec3 s4 = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o1.x, -o1.y), 0.0).rgb;
        float fineBlur = 0.25 * (dot(s1, lw) + dot(s2, lw) + dot(s3, lw) + dot(s4, lw));

        vec3 w1 = textureLod(ps0, in_TEXCOORD0.xy + vec2( o2.x,  0.0), 0.0).rgb;
        vec3 w2 = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o2.x,  0.0), 0.0).rgb;
        vec3 w3 = textureLod(ps0, in_TEXCOORD0.xy + vec2( 0.0,  o2.y), 0.0).rgb;
        vec3 w4 = textureLod(ps0, in_TEXCOORD0.xy + vec2( 0.0, -o2.y), 0.0).rgb;
        float wideLuma = 0.25 * (dot(w1, lw) + dot(w2, lw) + dot(w3, lw) + dot(w4, lw));

        float cLuma = dot(color.xyz, lw);
        float midProtect = 1.0 - smoothstep(0.38, 0.96, abs(cLuma * 2.0 - 1.0));
        float fineDetail = (cLuma - fineBlur) * (uClarity * 0.75 + uMicroRelief * 0.95) * (0.45 + 0.55 * midProtect);
        float medDetail = (cLuma - wideLuma) * (uMicroRelief * 0.45) * (0.35 + 0.65 * midProtect);
        color.xyz = clamp(color.xyz + vec3(clamp(fineDetail + medDetail, -0.22, 0.22)), 0.0, 1.0);

        if (uHdr > 0.001) {
            float localGain = clamp(pow(0.45 / (wideLuma + 0.05), 0.42), 0.90, 1.48);
            color.xyz *= mix(1.0, localGain, uHdr * 0.45);
            float hi = smoothstep(0.62, 1.0, dot(color.xyz, lw));
            color.xyz += color.xyz * hi * uHdr * 0.26;
        }

        if (uGlow > 0.001) {
            vec3 bloom = (max(w1 - 0.62, 0.0) + max(w2 - 0.62, 0.0) + max(w3 - 0.62, 0.0) + max(w4 - 0.62, 0.0)) * 0.25;
            color.xyz += bloom * uGlow * 1.45;
        }
    }

    if (uDehaze > 0.001) {
        float minCh = min(color.x, min(color.y, color.z));
        float hazeAmt = uDehaze * 0.14 * smoothstep(0.18, 0.75, minCh);
        color.xyz = clamp((color.xyz - hazeAmt) / max(1.0 - hazeAmt, 0.65), 0.0, 1.0);
    }

    if (uShadowLift > 0.001) {
        float lSh = dot(color.xyz, vec3(0.2126, 0.7152, 0.0722));
        float shMask = (1.0 - smoothstep(0.04, 0.42, lSh)) * smoothstep(0.005, 0.08, lSh);
        color.xyz = clamp(color.xyz + color.xyz * shMask * uShadowLift * 0.85 + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
    }

    color.xyz *= uExposure;
    if (uFilmic > 0.001) {
        vec3 mapped = acesFilmic(color.xyz * 1.18);
        color.xyz = mix(clamp(color.xyz, 0.0, 1.0), mapped, uFilmic);
    }
    color.xyz = clamp((color.xyz - 0.5) * uContrast + 0.5, 0.0, 1.0);

    if (uGrade > 0.001) {
        float lG = dot(color.xyz, vec3(0.2126, 0.7152, 0.0722));
        float sw = 1.0 - smoothstep(0.0, 0.52, lG);
        float hw = smoothstep(0.45, 1.0, lG);
        color.xyz = clamp(color.xyz + uGrade * (vec3(-0.03, 0.025, 0.07) * sw + vec3(0.07, 0.025, -0.045) * hw), 0.0, 1.0);
    }

    if (abs(uSat - 1.0) > 0.001) {
        float lSat = dot(color.xyz, vec3(0.2126, 0.7152, 0.0722));
        color.xyz = clamp(mix(vec3(lSat), color.xyz, uSat), 0.0, 1.0);
    }

    if (uBlack > 0.001) {
        float bp = uBlack * 0.06;
        color.xyz = clamp(max(color.xyz - bp, 0.0) / max(1.0 - bp, 0.5), 0.0, 1.0);
    }

    if (uVig > 0.001) {
        color.xyz *= 1.0 - uVig * smoothstep(0.32, 0.85, length(in_TEXCOORD0.xy - 0.5));
    }

    if (uLutAmt > 0.0) {
        highp vec3 lutCoord = color.xyz * uLutMap.x + uLutMap.y;
        color.xyz = mix(color.xyz, texture(uLut, lutCoord).rgb, uLutAmt);
    }

    color.w = 1.0;
    out_Target0 = color;
}
`;

// Fast path: Catmull-Rom 2x upscale (9 bilinear fetches), deringing clamp, adaptive sharpen, HyperReal LUT.
export const UPSCALE_FRAGMENT_SRC = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uSharp;
uniform float uMicroRelief;
uniform float uDehaze;
uniform float uShadowLift;
uniform float uHdr;
uniform float uFilmic;
uniform float uExposure;
uniform float uContrast;
uniform float uSat;
uniform float uGrade;
uniform float uBlack;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

in vec2 vUv;
out vec4 outColor;

vec3 tex(vec2 p) { return texture(uTex, p).rgb; }
vec3 acesFilmic(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

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
    vec3 col = clamp(c + (c - avg) * (uSharp + uMicroRelief * 0.45) * amp * 2.0, 0.0, 1.0);

    if (uDehaze > 0.001) {
        float minCh = min(col.r, min(col.g, col.b));
        float hazeAmt = uDehaze * 0.14 * smoothstep(0.18, 0.75, minCh);
        col = clamp((col - hazeAmt) / max(1.0 - hazeAmt, 0.65), 0.0, 1.0);
    }
    if (uShadowLift > 0.001) {
        float lSh = dot(col, vec3(0.2126, 0.7152, 0.0722));
        float shMask = (1.0 - smoothstep(0.04, 0.42, lSh)) * smoothstep(0.005, 0.08, lSh);
        col = clamp(col + col * shMask * uShadowLift * 0.8 + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
    }
    col *= uExposure;
    if (uFilmic > 0.001) col = mix(clamp(col, 0.0, 1.0), acesFilmic(col * 1.18), uFilmic);
    if (uHdr > 0.001) {
        float lc = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = clamp(col + col * smoothstep(0.62, 1.0, lc) * uHdr * 0.25, 0.0, 1.0);
    }
    col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);
    if (uGrade > 0.001) {
        float lG = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = clamp(col + uGrade * (vec3(-0.03, 0.025, 0.07) * (1.0 - smoothstep(0.0, 0.52, lG)) + vec3(0.07, 0.025, -0.045) * smoothstep(0.45, 1.0, lG)), 0.0, 1.0);
    }
    if (abs(uSat - 1.0) > 0.001) {
        float lSat = dot(col, vec3(0.2126, 0.7152, 0.0722));
        col = clamp(mix(vec3(lSat), col, uSat), 0.0, 1.0);
    }
    if (uBlack > 0.001) {
        float bp = uBlack * 0.06;
        col = clamp(max(col - bp, 0.0) / max(1.0 - bp, 0.5), 0.0, 1.0);
    }
    if (uLutAmt > 0.0) col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
    outColor = vec4(col, 1.0);
}
`;

// Full Look Filters pass: exact port of GlPipeline.kt FRAGMENT_SRC.
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
uniform float uMicroRelief;
uniform float uDehaze;
uniform float uShadowLift;
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

vec3 acesFilmicLook(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

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

    // Contrast-adaptive sharpening
    vec3 mn = min(e, min(min(n, s), min(w, ea)));
    vec3 mx = max(e, max(max(n, s), max(w, ea)));
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    float peak = mix(-0.125, -0.2, uSharp);
    vec3 wgt = amp * peak;
    vec3 sharp = (e + (n + s + w + ea) * wgt) / (1.0 + 4.0 * wgt);
    vec3 col = mix(e, sharp, step(0.001, uSharp));

    // Clarity + Micro-texture relief: dual-radius unsharp with midtone protection
    vec3 blur = (tap(vec2(-2.0, 0.0)) + tap(vec2(2.0, 0.0)) + tap(vec2(0.0, -2.0)) + tap(vec2(0.0, 2.0))) * 0.25;
    float midMask = 1.0 - abs(luma(col) * 2.0 - 1.0);
    col += (col - blur) * (uClarity + uMicroRelief * 0.65) * (0.5 + 0.5 * midMask);
    col = clamp(col, 0.0, 1.0);

    // Atmospheric dehaze & tactical shadow detail recovery
    if (uDehaze > 0.001) {
        float minCh = min(col.r, min(col.g, col.b));
        float hazeAmt = uDehaze * 0.14 * smoothstep(0.18, 0.75, minCh);
        col = clamp((col - hazeAmt) / max(1.0 - hazeAmt, 0.65), 0.0, 1.0);
    }
    if (uShadowLift > 0.001) {
        float lSh = luma(col);
        float shMask = (1.0 - smoothstep(0.04, 0.42, lSh)) * smoothstep(0.005, 0.08, lSh);
        col = clamp(col + col * shMask * uShadowLift * 0.85 + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
    }

    // Glow: inner 8-tap + optional wide 8-tap HDR bloom
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

    // HDR-style local tone mapping + highlight punch
    if (uHdr > 0.001) {
        vec3 wide = (tap(vec2(-14.0, -14.0)) + tap(vec2(14.0, -14.0)) + tap(vec2(-14.0, 14.0)) + tap(vec2(14.0, 14.0))
                   + tap(vec2(-26.0, 0.0)) + tap(vec2(26.0, 0.0)) + tap(vec2(0.0, -26.0)) + tap(vec2(0.0, 26.0))) * 0.125;
        float localL = luma(wide);
        float gain = clamp(pow(0.42 / (localL + 0.04), 0.5), 0.92, 1.7);
        col *= mix(1.0, gain, uHdr * 0.55);
        float lc = luma(col);
        float hi = smoothstep(0.62, 1.0, lc);
        col += col * hi * uHdr * 0.30;
        col = mix(col, col / (1.0 + max(col - 0.85, 0.0) * 1.2), uHdr);
    }

    // Exposure + ACES Filmic curve + contrast
    col *= uExposure;
    vec3 curved = acesFilmicLook(col * 1.18);
    col = mix(clamp(col, 0.0, 1.0), curved, uFilmic);
    col = (col - 0.5) * uContrast + 0.5;

    // Split-tone teal shadows / warm orange highlights
    float l0 = luma(col);
    float shadowsW = 1.0 - smoothstep(0.0, 0.55, l0);
    float highsW = smoothstep(0.45, 1.0, l0);
    col += uGrade * (vec3(-0.03, 0.03, 0.08) * shadowsW + vec3(0.08, 0.03, -0.05) * highsW);

    // Vibrance then saturation
    float chromaMax = max(col.r, max(col.g, col.b));
    float chromaMin = min(col.r, min(col.g, col.b));
    float satNow = chromaMax - chromaMin;
    col = mix(vec3(luma(col)), col, 1.0 + uVibrance * (1.0 - satNow));
    col = mix(vec3(luma(col)), col, uSat);

    // Deeper blacks / dehaze
    float bp = uBlack * 0.07;
    col = max(col - bp, 0.0) / (1.0 - bp);

    col *= 1.0 - uVig * smoothstep(0.30, 0.80, length(suv - 0.5));
    col = clamp(col, 0.0, 1.0);
    if (uLutAmt > 0.0) col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
    if (uMotion > 0.0 && uHasPrev > 0.5) col = mix(col, texture(uPrev, suv).rgb, uMotion);
    outColor = vec4(col, 1.0);
}
`;
