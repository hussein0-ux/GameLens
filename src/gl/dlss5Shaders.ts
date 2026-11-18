export const GBUFFER_VERTEX_SRC = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec4 aMat; // rgb = base albedo tint, a = roughness (0..1)

uniform mat4 uModel;
uniform mat4 uMvp;
uniform mat4 uPrevMvp;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;
out vec4 vMat;
out vec4 vCurrClip;
out vec4 vPrevClip;

void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(uModel) * aNormal;
    vUv = aUv;
    vMat = aMat;
    vCurrClip = uMvp * vec4(aPos, 1.0);
    vPrevClip = uPrevMvp * vec4(aPos, 1.0);
    gl_Position = vCurrClip;
}
`;

export const GBUFFER_FRAGMENT_SRC = `#version 300 es
precision highp float;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
in vec4 vMat;
in vec4 vCurrClip;
in vec4 vPrevClip;

uniform float uTime;
uniform float uObjId;
uniform float uFloatGbuf;
uniform vec2 uJitter;
uniform vec2 uRenderRes;

layout(location = 0) out vec4 gColor;
layout(location = 1) out vec4 gMotionDepth;
layout(location = 2) out vec4 gNormalMat;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 sunDir = normalize(vec3(0.55, 0.72, 0.42));
    vec3 skyDir = vec3(0.0, 1.0, 0.0);

    // High-frequency procedural surface detail so half-res aliasing vs 2x TAAU reconstruction is clearly visible
    vec2 tileUv = vWorldPos.xz * 2.0 + vUv * 4.0;
    float checker = step(0.5, fract(tileUv.x)) * 0.14 + step(0.5, fract(tileUv.y)) * 0.14;
    float fineLines = smoothstep(0.92, 0.98, abs(sin(vWorldPos.x * 18.0))) * 0.18
                    + smoothstep(0.92, 0.98, abs(sin(vWorldPos.z * 18.0))) * 0.18;
    float grain = (hash21(floor(vWorldPos.xz * 24.0 + vWorldPos.y * 12.0)) - 0.5) * 0.08;

    vec3 albedo = clamp(vMat.rgb * (0.88 + checker - fineLines * 0.45 + grain), 0.04, 1.0);
    // Micro-bump normal perturbation for crisp specular micro-facets
    float bX = sin(vWorldPos.x * 22.0 + vWorldPos.z * 6.0) * 0.08 + grain * 0.9;
    float bZ = cos(vWorldPos.z * 22.0 - vWorldPos.x * 6.0) * 0.08 - grain * 0.9;
    N = normalize(N + vec3(bX, 0.0, bZ) * (0.4 + 0.6 * vMat.a));

    // Drone energy core strip
    if (uObjId > 0.75) {
        float pulse = 0.65 + 0.35 * sin(uTime * 5.0);
        float strip = smoothstep(0.15, 0.0, abs(vNormal.y));
        albedo = mix(albedo, vec3(0.15, 0.82, 1.0) * (1.2 + pulse * 0.6), strip * 0.6);
    }

    float ndl = max(dot(N, sunDir), 0.0);
    float sky = 0.5 + 0.5 * dot(N, skyDir);
    vec3 lit = albedo * (vec3(1.24, 1.08, 0.90) * ndl + vec3(0.32, 0.40, 0.52) * sky + 0.10);

    // Fog atmosphere
    float linearDepth = clamp(vCurrClip.w / 60.0, 0.0, 0.98);
    vec3 fogCol = vec3(0.68, 0.74, 0.82);
    lit = mix(lit, fogCol, smoothstep(0.35, 0.98, linearDepth) * 0.5);

    gColor = vec4(clamp(lit, 0.0, 1.0), 1.0);

    // Exact unjittered screen-space motion vector (currUv - prevUv)
    vec2 currNdc = vCurrClip.xy / max(vCurrClip.w, 1e-4);
    vec2 prevNdc = vPrevClip.xy / max(vPrevClip.w, 1e-4);
    vec2 jitterNdc = (2.0 * uJitter) / max(uRenderRes, vec2(1.0));
    vec2 mv = ((currNdc + jitterNdc) - prevNdc) * 0.5;

    if (uFloatGbuf > 0.5) {
        gMotionDepth = vec4(mv, linearDepth, uObjId);
    } else {
        gMotionDepth = vec4(clamp(mv * 8.0 + 0.5, 0.0, 1.0), linearDepth, uObjId);
    }

    gNormalMat = vec4(N * 0.5 + 0.5, vMat.a);
}
`;

export const DLSS5_NEURAL_TAAU_FS = `#version 300 es
precision highp float;

uniform sampler2D uGColor;
uniform sampler2D uGMotionDepth;
uniform sampler2D uGNormalMat;
uniform sampler2D uHistoryPrev;

uniform vec2 uRenderRes;
uniform vec2 uOutRes;
uniform vec2 uJitter;
uniform float uFloatGbuf;
uniform float uHasHistory;
uniform float uNeuralLight; // 0..1 3D-guided neural lighting & material reconstruction
uniform int uTier;          // 1 = Neural 3D-Guided + TAAU, 2 = SGSR2 TAAU, 3 = Spatial fallback

in vec2 vUv;
out vec4 outColor;

vec2 decodeMv(vec4 md) {
    return uFloatGbuf > 0.5 ? md.xy : (md.xy - 0.5) / 8.0;
}

vec3 rgbToYcocg(vec3 c) {
    float y  = dot(c, vec3(0.25, 0.5, 0.25));
    float co = dot(c, vec3(0.5, 0.0, -0.5));
    float cg = dot(c, vec3(-0.25, 0.5, -0.25));
    return vec3(y, co, cg);
}

vec3 ycocgToRgb(vec3 ycc) {
    float t = ycc.x - ycc.z;
    return clamp(vec3(t + ycc.y, ycc.x + ycc.z, t - ycc.y), 0.0, 1.0);
}

// 3D-Guided Neural Lighting & Material Synthesis:
// Uses G-buffer depth, surface normals, and roughness with strict depth-discontinuity stopping.
vec3 applyNeural3dLighting(vec2 uv, vec3 baseCol, vec4 md, vec4 nm) {
    float depth0 = md.z;
    vec3 N0 = normalize(nm.rgb * 2.0 - 1.0);
    float roughness = nm.a;
    vec2 texel = 1.0 / uRenderRes;

    // Depth-guided screen-space contact AO + directional bounce
    float ao = 0.0;
    vec3 bounce = vec3(0.0);
    const vec2 dirs[6] = vec2[6](
        vec2(1.0, 0.0), vec2(0.5, 0.866), vec2(-0.5, 0.866),
        vec2(-1.0, 0.0), vec2(-0.5, -0.866), vec2(0.5, -0.866)
    );
    for (int i = 0; i < 6; i++) {
        vec2 suv = clamp(uv + dirs[i] * texel * 3.5, vec2(0.0), vec2(1.0));
        float dTap = texture(uGMotionDepth, suv).z;
        vec3 cTap = texture(uGColor, suv).rgb;
        vec3 nTap = normalize(texture(uGNormalMat, suv).rgb * 2.0 - 1.0);

        float depthDiff = depth0 - dTap;
        float rangeStop = exp(-abs(depthDiff) * 55.0);
        // Occluder closer than center darkens crease
        float occ = smoothstep(0.002, 0.025, depthDiff) * rangeStop;
        ao += occ;

        // Warm indirect bounce from nearby sunlit surfaces facing each other
        float facing = max(0.0, dot(N0, -nTap) * 0.5 + 0.5);
        bounce += cTap * facing * rangeStop;
    }
    ao = 1.0 - clamp((ao / 6.0) * 1.4, 0.0, 0.65);
    bounce = (bounce / 6.0) * 0.35;

    // View-space approximate specular / Fresnel rim grounded to material roughness
    vec3 V = normalize(vec3((0.5 - uv.x) * 1.2, (0.5 - uv.y) * 0.8, 1.0));
    vec3 sunDir = normalize(vec3(0.55, 0.72, 0.42));
    vec3 H = normalize(sunDir + V);
    float specPow = mix(96.0, 8.0, roughness);
    float spec = pow(max(dot(N0, H), 0.0), specPow) * (1.0 - roughness * 0.85);
    float fresnel = pow(1.0 - max(dot(N0, V), 0.0), 4.0) * (1.0 - roughness * 0.7);

    // Do not apply lighting synthesis to background sky pixels (depth == 1.0 or objId == 0.0)
    float skyMask = step(0.05, md.a) * (1.0 - smoothstep(0.90, 0.99, depth0));
    vec3 enhanced = baseCol * mix(1.0, ao, 0.85 * skyMask)
                  + bounce * skyMask
                  + vec3(1.0, 0.90, 0.72) * spec * 0.48 * skyMask
                  + vec3(0.45, 0.68, 0.95) * fresnel * 0.22 * skyMask;

    return clamp(mix(baseCol, enhanced, uNeuralLight * skyMask), 0.0, 1.0);
}

void main() {
    vec2 texel = 1.0 / uRenderRes;
    // Unjittered lookup into current half-res G-buffer (proj[8/9] shifts NDC by -jitterNdc)
    vec2 unjitteredUv = clamp(vUv + (uJitter / uRenderRes), vec2(0.0), vec2(1.0));
    vec4 mdCenter = texture(uGMotionDepth, unjitteredUv);
    vec4 nmCenter = texture(uGNormalMat, unjitteredUv);
    vec3 currCol = texture(uGColor, unjitteredUv).rgb;

    if (uTier == 1 && uNeuralLight > 0.01) {
        currCol = applyNeural3dLighting(unjitteredUv, currCol, mdCenter, nmCenter);
    }

    if (uTier == 3 || uHasHistory < 0.5) {
        outColor = vec4(currCol, 1.0);
        return;
    }

    // Closest-depth 3x3 velocity dilation for clean foreground silhouette edges
    vec2 bestMv = decodeMv(mdCenter);
    float bestDepth = mdCenter.z;
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offUv = clamp(unjitteredUv + vec2(float(x), float(y)) * texel, vec2(0.0), vec2(1.0));
            vec4 mdTap = texture(uGMotionDepth, offUv);
            if (mdTap.z < bestDepth) {
                bestDepth = mdTap.z;
                bestMv = decodeMv(mdTap);
            }
            vec3 cTap = texture(uGColor, offUv).rgb;
            if (uTier == 1 && uNeuralLight > 0.01 && (x != 0 || y != 0)) {
                cTap = applyNeural3dLighting(offUv, cTap, mdTap, texture(uGNormalMat, offUv));
            }
            vec3 ycc = rgbToYcocg(cTap);
            m1 += ycc;
            m2 += ycc * ycc;
        }
    }

    vec3 meanYcc = m1 / 9.0;
    vec3 stdYcc = sqrt(max(m2 / 9.0 - meanYcc * meanYcc, vec3(0.0)));
    vec3 boxMin = meanYcc - stdYcc * 1.25;
    vec3 boxMax = meanYcc + stdYcc * 1.25;

    vec2 prevUv = vUv - bestMv;
    if (prevUv.x < 0.0 || prevUv.x > 1.0 || prevUv.y < 0.0 || prevUv.y > 1.0) {
        outColor = vec4(currCol, 1.0);
        return;
    }

    vec3 histRgb = texture(uHistoryPrev, prevUv).rgb;
    vec3 histYcc = clamp(rgbToYcocg(histRgb), boxMin, boxMax);
    vec3 clampedHist = ycocgToRgb(histYcc);

    // Higher blend alpha when velocity is fast to stay crisp
    float mvSpeed = length(bestMv * uOutRes);
    float alpha = clamp(0.10 + mvSpeed * 0.025, 0.10, 0.35);
    outColor = vec4(mix(clampedHist, currCol, alpha), 1.0);
}
`;

export const DLSS5_FRAMEGEN_FS = `#version 300 es
precision highp float;

uniform sampler2D uHistoryPrev;
uniform sampler2D uHistoryCurr;
uniform sampler2D uGMotionDepth;
uniform float uFloatGbuf;
uniform float uBlendT; // 0.5 for midpoint interpolated frame

in vec2 vUv;
out vec4 outColor;

vec2 decodeMv(vec4 md) {
    return uFloatGbuf > 0.5 ? md.xy : (md.xy - 0.5) / 8.0;
}

void main() {
    vec2 gTexel = 1.0 / vec2(textureSize(uGMotionDepth, 0));
    vec4 mdBest = texture(uGMotionDepth, vUv);
    float bestDepth = mdBest.z;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec4 tap = texture(uGMotionDepth, clamp(vUv + vec2(float(x), float(y)) * gTexel * 2.0, vec2(0.0), vec2(1.0)));
            if (tap.z < bestDepth) {
                bestDepth = tap.z;
                mdBest = tap;
            }
        }
    }
    vec2 mv = decodeMv(mdBest);

    // Bidirectional warp halfway along dilated foreground motion vector
    vec2 uvPrev = clamp(vUv - mv * (1.0 - uBlendT), vec2(0.0), vec2(1.0));
    vec2 uvCurr = clamp(vUv + mv * uBlendT, vec2(0.0), vec2(1.0));

    vec3 cPrev = texture(uHistoryPrev, uvPrev).rgb;
    vec3 cCurr = texture(uHistoryCurr, uvCurr).rgb;
    vec3 cUnwarpedCurr = texture(uHistoryCurr, vUv).rgb;

    // Strict disocclusion rejection: reject warped prev on silhouette discontinuities to avoid ghosting
    float diff = length(cPrev - cCurr);
    float conf = smoothstep(0.16, 0.03, diff);
    vec3 blended = mix(cPrev, cCurr, uBlendT);
    vec3 col = mix(cUnwarpedCurr, blended, conf);

    outColor = vec4(col, 1.0);
}
`;

export const DLSS5_PRESENT_FS = `#version 300 es
precision highp float;

uniform sampler2D uTex;
uniform vec2 uOutRes;
uniform float uSharp;
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
uniform float uBypass;
uniform float uShowHud;
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

in vec2 vUv;
out vec4 outColor;

vec3 acesFilmicP(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec2 texel = 1.0 / uOutRes;
    vec3 col = texture(uTex, vUv).rgb;

    if (uBypass < 0.5) {
        // Native-resolution CAS sharpening + micro-texture relief
        vec3 n = texture(uTex, vUv + vec2(0.0, -texel.y)).rgb;
        vec3 s = texture(uTex, vUv + vec2(0.0,  texel.y)).rgb;
        vec3 w = texture(uTex, vUv + vec2(-texel.x, 0.0)).rgb;
        vec3 e = texture(uTex, vUv + vec2( texel.x, 0.0)).rgb;
        vec3 mn = min(col, min(min(n, s), min(w, e)));
        vec3 mx = max(col, max(max(n, s), max(w, e)));
        vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
        vec3 avg = (n + s + w + e) * 0.25;
        col = clamp(col + (col - avg) * (uSharp + uMicroRelief * 0.55) * amp * 1.6, 0.0, 1.0);

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
        if (uGlow > 0.001) {
            vec2 gOff = texel * 4.5;
            vec3 b1 = max(texture(uTex, vUv + vec2( gOff.x,  gOff.y)).rgb - 0.62, 0.0);
            vec3 b2 = max(texture(uTex, vUv + vec2(-gOff.x,  gOff.y)).rgb - 0.62, 0.0);
            vec3 b3 = max(texture(uTex, vUv + vec2( gOff.x, -gOff.y)).rgb - 0.62, 0.0);
            vec3 b4 = max(texture(uTex, vUv + vec2(-gOff.x, -gOff.y)).rgb - 0.62, 0.0);
            col = clamp(col + (b1 + b2 + b3 + b4) * 0.25 * uGlow * 1.35, 0.0, 1.0);
        }
        col *= uExposure;
        if (uFilmic > 0.001) col = mix(clamp(col, 0.0, 1.0), acesFilmicP(col * 1.18), uFilmic);
        if (uHdr > 0.001) {
            float lc = dot(col, vec3(0.2126, 0.7152, 0.0722));
            col = clamp(col + col * smoothstep(0.62, 1.0, lc) * uHdr * 0.24, 0.0, 1.0);
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

        if (uLutAmt > 0.0) {
            col = mix(col, texture(uLut, col * uLutMap.x + uLutMap.y).rgb, uLutAmt);
        }
    }

    // Native-res crisp HUD reticle (immune to TAAU/FrameGen ghosting, matching engine HUD mask)
    if (uShowHud > 0.5) {
        vec2 px = (vUv - 0.5) * uOutRes;
        float dist = length(px);
        float ring = smoothstep(1.4, 0.4, abs(dist - 14.0));
        float crossH = step(abs(px.y), 1.0) * step(4.0, abs(px.x)) * step(abs(px.x), 10.0);
        float crossV = step(abs(px.x), 1.0) * step(4.0, abs(px.y)) * step(abs(px.y), 10.0);
        float centerDot = smoothstep(2.0, 0.8, dist);
        float hudAlpha = clamp(max(max(ring * 0.7, centerDot), max(crossH, crossV) * 0.85), 0.0, 1.0);
        col = mix(col, vec3(0.95, 0.98, 1.0), hudAlpha * 0.85);
    }

    outColor = vec4(col, 1.0);
}
`;
