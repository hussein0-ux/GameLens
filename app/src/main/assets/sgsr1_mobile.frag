#version 310 es
// NOTE: Adapted for GameLens Snapdragon 8 Gen 3 (Adreno 750) 120 Hz HyperReal pipeline.
// Combines Qualcomm SGSR v1 edge-directed upscale with fused dual-band micro-relief, dark-channel dehaze,
// tactical shadow recovery, local HDR tone mapping, ACES Filmic, micro-bloom, grading, and HyperReal 3D LUT.
// Original SGSR1 portions: Snapdragon Game Super Resolution v1 (sgsr1_shader_mobile.frag), BSD-3-Clause.

//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

precision mediump float;
precision highp int;

////////////////////////
// USER CONFIGURATION //
////////////////////////

/*
* Operation modes:
* RGBA -> 1
* RGBY -> 3
* LERP -> 4
*/
#define OperationMode 1

#define EdgeThreshold 8.0/255.0

uniform float uEdgeSharpness;

// #define UseUniformBlock

////////////////////////
////////////////////////
////////////////////////

#if defined(UseUniformBlock)
layout (set=0, binding = 0) uniform UniformBlock
{
	highp vec4 ViewportInfo[1];
};
layout(set = 0, binding = 1) uniform mediump sampler2D ps0;
#else
uniform highp vec4 ViewportInfo[1];
uniform mediump sampler2D ps0;
#endif

// GameLens: HyperReal 3D LUT applied to the upscaled colour (same mapping as the other GameLens shaders).
uniform highp sampler3D uLut;
uniform float uLutAmt;
uniform vec2 uLutMap;

// GameLens: Fused 120 Hz Adreno 750 HyperReal controls (applied after SGSR edge reconstruction, before LUT).
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
layout(location=0) out vec4 out_Target0;

float fastLanczos2(float x)
{
	float wA = x-4.0;
	float wB = x*wA-wA;
	wA *= wA;
	return wB*wA;
}
vec2 weightY(float dx, float dy,float c, float std)
{
	float x = ((dx*dx)+(dy* dy))* 0.55 + clamp(abs(c)*std, 0.0, 1.0);
	float w = fastLanczos2(x);
	return vec2(w, w * c);
}

float luma709(vec3 c)
{
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 acesFilm(vec3 x)
{
	const float a = 2.51;
	const float b = 0.03;
	const float c = 2.43;
	const float d = 0.59;
	const float e = 0.14;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
	const int mode = OperationMode;
	float edgeThreshold = EdgeThreshold;
	float edgeSharpness = uEdgeSharpness;

	vec4 color;
	if(mode == 1)
		color.xyz = textureLod(ps0,in_TEXCOORD0.xy,0.0).xyz;
	else
		color.xyzw = textureLod(ps0,in_TEXCOORD0.xy,0.0).xyzw;

	//if ( mode!=4 && xCenter*xCenter+yCenter*yCenter<=0.4 * 0.4)
	if ( mode!=4)
	{
		highp vec2 imgCoord = ((in_TEXCOORD0.xy*ViewportInfo[0].zw)+vec2(-0.5,0.5));
		highp vec2 imgCoordPixel = floor(imgCoord);
		highp vec2 coord = (imgCoordPixel*ViewportInfo[0].xy);
		vec2 pl = (imgCoord+(-imgCoordPixel));
		vec4  left = textureGather(ps0,coord, mode);

		float edgeVote = abs(left.z - left.y) + abs(color[mode] - left.y)  + abs(color[mode] - left.z) ;
		if(edgeVote > edgeThreshold)
		{
			coord.x += ViewportInfo[0].x;

			vec4 right = textureGather(ps0,coord + highp vec2(ViewportInfo[0].x, 0.0), mode);
			vec4 upDown;
			upDown.xy = textureGather(ps0,coord + highp vec2(0.0, -ViewportInfo[0].y),mode).wz;
			upDown.zw  = textureGather(ps0,coord+ highp vec2(0.0, ViewportInfo[0].y), mode).yx;

			float mean = (left.y+left.z+right.x+right.w)*0.25;
			left = left - vec4(mean);
			right = right - vec4(mean);
			upDown = upDown - vec4(mean);
			color.w =color[mode] - mean;

			float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));
			float std = 2.181818/max(sum, 1e-4);

			vec2 aWY = weightY(pl.x, pl.y+1.0, upDown.x,std);
			aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y,std);
			aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z,std);
			aWY += weightY(pl.x, pl.y-2.0, upDown.w,std);
			aWY += weightY(pl.x+1.0, pl.y-1.0, left.x,std);
			aWY += weightY(pl.x, pl.y-1.0, left.y,std);
			aWY += weightY(pl.x, pl.y, left.z,std);
			aWY += weightY(pl.x+1.0, pl.y, left.w,std);
			aWY += weightY(pl.x-1.0, pl.y-1.0, right.x,std);
			aWY += weightY(pl.x-2.0, pl.y-1.0, right.y,std);
			aWY += weightY(pl.x-2.0, pl.y, right.z,std);
			aWY += weightY(pl.x-1.0, pl.y, right.w,std);

			float finalY = aWY.y/max(aWY.x, 1e-4);

			float maxY = max(max(left.y,left.z),max(right.x,right.w));
			float minY = min(min(left.y,left.z),min(right.x,right.w));
			finalY = clamp(edgeSharpness*finalY, minY, maxY);

			float deltaY = finalY -color.w;

			//smooth high contrast input
			deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

			color.x = clamp((color.x+deltaY),0.0,1.0);
			color.y = clamp((color.y+deltaY),0.0,1.0);
			color.z = clamp((color.z+deltaY),0.0,1.0);
		}
	}

	vec3 col = clamp(color.xyz, 0.0, 1.0);

	// Shared 8-tap spatial neighborhood (tight 1.0x cross + wide 2.75x diagonal) drives
	// atmospheric dehaze, dual-band micro-relief, clarity, tactical shadow lift, micro-bloom, and local HDR.
	if ((uClarity + uMicroRelief + uDehaze + uShadowLift + uHdr + uGlow) > 0.001)
	{
		highp vec2 o1 = ViewportInfo[0].xy;
		highp vec2 o2 = ViewportInfo[0].xy * 2.75;

		vec3 tN = textureLod(ps0, in_TEXCOORD0.xy + vec2(0.0, -o1.y), 0.0).xyz;
		vec3 tS = textureLod(ps0, in_TEXCOORD0.xy + vec2(0.0,  o1.y), 0.0).xyz;
		vec3 tW = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o1.x, 0.0), 0.0).xyz;
		vec3 tE = textureLod(ps0, in_TEXCOORD0.xy + vec2( o1.x, 0.0), 0.0).xyz;
		vec3 nearAvg = (tN + tS + tW + tE) * 0.25;

		vec3 tNW = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o2.x, -o2.y), 0.0).xyz;
		vec3 tNE = textureLod(ps0, in_TEXCOORD0.xy + vec2( o2.x, -o2.y), 0.0).xyz;
		vec3 tSW = textureLod(ps0, in_TEXCOORD0.xy + vec2(-o2.x,  o2.y), 0.0).xyz;
		vec3 tSE = textureLod(ps0, in_TEXCOORD0.xy + vec2( o2.x,  o2.y), 0.0).xyz;
		vec3 wideAvg = (tNW + tNE + tSW + tSE) * 0.25;

		// 1. Dark-channel atmospheric dehaze
		if (uDehaze > 0.001)
		{
			vec3 darkRgb = min(col, min(min(tNW, tNE), min(tSW, tSE)));
			float darkCh = min(darkRgb.r, min(darkRgb.g, darkRgb.b));
			float veil = clamp(darkCh * 0.42 * uDehaze, 0.0, 0.45);
			col = clamp((col - veil) / max(1.0 - veil, 0.55), 0.0, 1.0);
		}

		// 2. Dual-band micro-texture high-pass relief (suppresses shadow noise & highlight clipping)
		if (uMicroRelief > 0.001)
		{
			vec3 hpFine = col - nearAvg;
			vec3 hpMed = nearAvg - wideAvg;
			float lc0 = luma709(col);
			float reliefMask = clamp(4.0 * lc0 * (1.0 - lc0), 0.15, 1.0);
			col = clamp(col + (hpFine * 1.15 + hpMed * 0.55) * (uMicroRelief * reliefMask), 0.0, 1.0);
		}

		// 3. Midtone-protected local clarity
		if (uClarity > 0.001)
		{
			float lc1 = luma709(col);
			float blurLuma = luma709(wideAvg);
			float midMask = 1.0 - abs(lc1 * 2.0 - 1.0);
			col = clamp(col + vec3((lc1 - blurLuma) * uClarity * (0.5 + 0.5 * midMask)), 0.0, 1.0);
		}

		float localL = luma709(wideAvg);

		// 4. Tactical shadow detail recovery (lifts dark rooms/doorways while keeping bright areas untouched)
		if (uShadowLift > 0.001)
		{
			float lc2 = luma709(col);
			float shMask = (1.0 - smoothstep(0.0, 0.45, lc2)) * (1.0 - smoothstep(0.05, 0.55, localL));
			col = clamp(col + col * (shMask * uShadowLift * 0.90) + vec3(0.015 * shMask * uShadowLift), 0.0, 1.0);
		}

		// 5. Micro-bloom on specular highlights
		if (uGlow > 0.001)
		{
			vec3 bloom = max(wideAvg - 0.58, vec3(0.0)) + max(nearAvg - 0.65, vec3(0.0)) * 0.5;
			col = col + bloom * (uGlow * 1.25);
		}

		// 6. Local HDR tone mapping + specular punch + smooth highlight roll-off
		if (uHdr > 0.001)
		{
			float gain = clamp(sqrt(clamp(0.42 / (localL + 0.04), 0.0, 4.0)), 0.90, 1.65);
			col *= mix(1.0, gain, uHdr * 0.55);
			float lc3 = luma709(col);
			float hi = smoothstep(0.60, 1.0, lc3);
			col += col * hi * uHdr * 0.28;
			col = mix(col, col / (1.0 + max(col - 0.85, vec3(0.0)) * 1.2), uHdr);
		}
	}

	// 7. Exposure, ACES Filmic tone mapping, contrast
	col *= uExposure;
	if (uFilmic > 0.001)
	{
		col = mix(clamp(col, 0.0, 1.0), acesFilm(max(col, vec3(0.0))), uFilmic);
	}
	col = (col - 0.5) * uContrast + 0.5;

	// 8. Split-tone grade (cool teal shadows / warm amber highlights)
	if (uGrade > 0.001)
	{
		float l0 = luma709(col);
		float shadowsW = 1.0 - smoothstep(0.0, 0.55, l0);
		float highsW = smoothstep(0.45, 1.0, l0);
		col += uGrade * (vec3(-0.03, 0.03, 0.08) * shadowsW + vec3(0.08, 0.03, -0.05) * highsW);
	}

	// 9. Vibrance & saturation
	if (uVibrance > 0.001)
	{
		float satNow = max(col.x, max(col.y, col.z)) - min(col.x, min(col.y, col.z));
		float lv = luma709(col);
		col = vec3(lv) + (col - vec3(lv)) * (1.0 + uVibrance * (1.0 - satNow));
	}
	col = mix(vec3(luma709(col)), col, uSat);

	// 10. Black point anchor & vignette
	if (uBlack > 0.001)
	{
		float bp = uBlack * 0.07;
		col = max(col - bp, vec3(0.0)) / (1.0 - bp);
	}
	if (uVig > 0.001)
	{
		col *= 1.0 - uVig * smoothstep(0.30, 0.80, length(in_TEXCOORD0.xy - 0.5));
	}

	col = clamp(col, 0.0, 1.0);

	// 11. HyperReal Rec709 33^3 3D LUT
	if (uLutAmt > 0.0)
	{
		highp vec3 lutCoord = col * uLutMap.x + uLutMap.y;
		col = mix(col, texture(uLut, lutCoord).rgb, uLutAmt);
	}

	out_Target0 = vec4(col, 1.0);
}