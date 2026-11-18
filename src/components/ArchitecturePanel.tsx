import React from 'react';
import { PRESETS } from '../engine/settings';

export const ArchitecturePanel: React.FC = () => {
  return (
    <div className="flex flex-col gap-6">
      <section className="rounded-lg border border-[#232C3D] bg-[#121721] p-6">
        <h2 className="font-display text-lg font-semibold text-slate-100">
          Real-Time GPU Post-Processing Architecture
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-300">
          GameLens is a screen-space post-processor ported from OpenGL ES 3.0/3.1 to WebGL2.
          It stays on the GPU across every stage: captured game frames or uploaded clips are optionally
          downsampled to half screen resolution, edge-anti-aliased, upscaled 2x via Qualcomm Snapdragon
          Game Super Resolution 1 or Catmull-Rom bicubic filtering, and graded through a 33³ HyperReal
          Rec.709 3D lookup table.
        </p>

        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-md border border-[#232C3D] bg-[#0F141D] p-4">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              01. Qualcomm SGSR 1 + FXAA 3
            </h3>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
              Pre-pass FXAA 3 Console smooths high-contrast staircase edges without blurring low-contrast
              surface textures, followed by edge-directed 12-tap Lanczos2 reconstruction, local vibrance,
              and 4-tap luma clarity.
            </p>
          </div>
          <div className="rounded-md border border-[#232C3D] bg-[#0F141D] p-4">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              02. Catmull-Rom 2x + CAS
            </h3>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
              Nine bilinear fetches reconstruct bicubic detail, clamped against the nearest 4-neighbor
              luma envelope to eliminate ringing halos, then finished with Contrast-Adaptive Sharpening.
            </p>
          </div>
          <div className="rounded-md border border-[#232C3D] bg-[#0F141D] p-4">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              03. Full Look &amp; 33³ 3D LUT
            </h3>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
              Includes local HDR tone mapping with highlight roll-off, two-ring bloom glow, teal/orange
              split toning, deep blacks dehaze, chromatic fringe, ping-pong FBO motion blur, and Adobe
              <code className="font-mono">.cube</code> 3D texture grading.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-[#232C3D] bg-[#121721] p-6">
        <h2 className="font-display text-base font-semibold text-slate-100">
          Preset Calibration Matrix
        </h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#232C3D] text-slate-400">
                <th className="py-2.5 pr-4 font-medium">Preset</th>
                <th className="py-2.5 px-3 font-medium text-right">Sharpen</th>
                <th className="py-2.5 px-3 font-medium text-right">Clarity</th>
                <th className="py-2.5 px-3 font-medium text-right">Contrast</th>
                <th className="py-2.5 px-3 font-medium text-right">Saturation</th>
                <th className="py-2.5 px-3 font-medium text-right">HDR</th>
                <th className="py-2.5 px-3 font-medium text-right">Grade</th>
                <th className="py-2.5 pl-3 font-medium text-right">Glow</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1D2535] font-mono">
              {PRESETS.map((p) => (
                <tr key={p.id} className="hover:bg-[#161D2B]">
                  <td className="py-2.5 pr-4 font-sans font-medium text-slate-100">{p.label}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.sharpen.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.clarity.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.contrast.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.saturation.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.hdr.toFixed(2)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{p.grade.toFixed(2)}</td>
                  <td className="py-2.5 pl-3 text-right text-slate-300">{p.glow.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-lg border border-[#232C3D] bg-[#121721] p-6 text-xs leading-relaxed text-slate-400">
        <h3 className="font-display text-sm font-semibold text-slate-200">
          Snapdragon Game Super Resolution Attribution (BSD-3-Clause)
        </h3>
        <p className="mt-2">
          Copyright (c) 2023–2025, Qualcomm Innovation Center, Inc. All rights reserved. Adapted for
          WebGL2 (`#version 300 es`) from <code className="font-mono">sgsr1_mobile.frag</code>. See{' '}
          <a
            href="/assets/SGSR_LICENSE.txt"
            target="_blank"
            rel="noreferrer"
            className="text-blue-400 underline hover:text-blue-300"
          >
            SGSR_LICENSE.txt
          </a>{' '}
          for full license terms.
        </p>
      </section>
    </div>
  );
};
