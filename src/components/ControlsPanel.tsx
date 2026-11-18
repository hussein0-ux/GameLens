import React from 'react';
import {
  LOOK_SLIDERS,
  MODE_OPTIONS,
  PRESETS,
  UPSCALER_OPTIONS,
  UPSCALER_SLIDERS,
  type LookPreset,
  type LookSettingsState,
  type PipelineMode,
  type UpscalerType,
} from '../engine/settings';

interface ControlsPanelProps {
  settings: LookSettingsState;
  activePresetId: string | null;
  onUpdateSetting: <K extends keyof LookSettingsState>(key: K, value: LookSettingsState[K]) => void;
  onApplyPreset: (preset: LookPreset) => void;
  onResetDefaults: () => void;
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({
  settings,
  activePresetId,
  onUpdateSetting,
  onApplyPreset,
  onResetDefaults,
}) => {
  return (
    <aside className="flex flex-col gap-6 rounded-lg border border-[#232C3D] bg-[#121721] p-5">
      {/* Presets Section */}
      <div>
        <div className="flex items-center justify-between">
          <h2 className="font-display text-sm font-semibold text-slate-100">Look Presets</h2>
          <button
            type="button"
            onClick={onResetDefaults}
            className="text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
          >
            Reset defaults
          </button>
        </div>
        <div className="mt-3 grid grid-cols-3 gap-2">
          {PRESETS.map((preset) => {
            const active = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => onApplyPreset(preset)}
                title={preset.summary}
                className={`rounded-md border px-3 py-2 text-xs font-medium whitespace-nowrap transition-colors ${
                  active
                    ? 'border-blue-500 bg-blue-600 text-white'
                    : 'border-[#252F42] bg-[#181F2C] text-slate-200 hover:border-slate-600 hover:bg-[#1E2738]'
                }`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>
      </div>

      <hr className="border-[#232C3D]" />

      {/* Pipeline Mode */}
      <div>
        <h2 className="font-display text-sm font-semibold text-slate-100">Mode</h2>
        <div className="mt-2.5 flex flex-col gap-2">
          {MODE_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={`flex cursor-pointer items-start gap-2.5 rounded-md border px-3 py-2 text-xs transition-colors ${
                settings.mode === opt.value
                  ? 'border-blue-500/60 bg-blue-600/10 text-slate-100'
                  : 'border-[#232C3D] bg-[#161C28] text-slate-300 hover:border-slate-600'
              }`}
            >
              <input
                type="radio"
                name="pipeline-mode"
                checked={settings.mode === opt.value}
                onChange={() => onUpdateSetting('mode', opt.value as PipelineMode)}
                className="mt-0.5 accent-blue-500"
              />
              <span>{opt.label}</span>
            </label>
          ))}
        </div>

        <div className="mt-3 flex flex-col gap-2">
          <label className="flex cursor-pointer items-center gap-2.5 text-xs text-slate-200">
            <input
              type="checkbox"
              checked={settings.captureHalf}
              onChange={(e) => onUpdateSetting('captureHalf', e.target.checked)}
              className="h-3.5 w-3.5 rounded accent-blue-500"
            />
            <span>Capture at half screen size (applies on next start)</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2.5 text-xs text-slate-200">
            <input
              type="checkbox"
              checked={settings.bypass}
              onChange={(e) => onUpdateSetting('bypass', e.target.checked)}
              className="h-3.5 w-3.5 rounded accent-blue-500"
            />
            <span>Bypass look (raw capture-rate test)</span>
          </label>
        </div>
      </div>

      <hr className="border-[#232C3D]" />

      {/* GPU Upscaler & Enhance */}
      <div>
        <h2 className="font-display text-sm font-semibold text-slate-100">GPU Upscaler</h2>
        <div className="mt-2.5 flex flex-col gap-2">
          {UPSCALER_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={`flex cursor-pointer items-center gap-2.5 rounded-md border px-3 py-2 text-xs transition-colors ${
                settings.upscaler === opt.value
                  ? 'border-blue-500/60 bg-blue-600/10 text-slate-100'
                  : 'border-[#232C3D] bg-[#161C28] text-slate-300 hover:border-slate-600'
              }`}
            >
              <input
                type="radio"
                name="gpu-upscaler"
                checked={settings.upscaler === opt.value}
                onChange={() => onUpdateSetting('upscaler', opt.value as UpscalerType)}
                className="accent-blue-500"
              />
              <span>{opt.label}</span>
            </label>
          ))}
        </div>

        <div className="mt-4 flex flex-col gap-3">
          {UPSCALER_SLIDERS.map((s) => {
            const val = settings[s.key] as number;
            return (
              <div key={s.key} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300">{s.label}</span>
                  <span className="font-mono text-slate-200">{val.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={val}
                  onChange={(e) => onUpdateSetting(s.key, Number.parseFloat(e.target.value))}
                  className="gamelens-slider"
                />
              </div>
            );
          })}
        </div>
      </div>

      <hr className="border-[#232C3D]" />

      {/* Look & Grade Sliders */}
      <div>
        <div className="flex items-center justify-between">
          <h2 className="font-display text-sm font-semibold text-slate-100">Look &amp; Grade</h2>
          <span className="text-[11px] text-slate-400">
            {settings.mode === 0 ? 'All filters active' : 'LUT active in upscale modes'}
          </span>
        </div>
        <div className="mt-3 flex flex-col gap-3">
          {LOOK_SLIDERS.map((s) => {
            const val = settings[s.key] as number;
            return (
              <div key={s.key} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300">{s.label}</span>
                  <span className="font-mono text-slate-200">{val.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={val}
                  onChange={(e) => onUpdateSetting(s.key, Number.parseFloat(e.target.value))}
                  className="gamelens-slider"
                />
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
};
