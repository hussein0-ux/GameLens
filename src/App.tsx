import { ChangeEvent, useCallback, useEffect, useRef, useState } from 'react';
import { Box, Copy, Monitor, Play, RotateCcw, Square, Upload, Zap } from 'lucide-react';
import { clearCrashLog, readCrashLog, recordCrash } from './core/crashLog';
import {
  applyPreset,
  DEFAULT_LOOK_STATE,
  LOOK_SLIDERS,
  LookState,
  MODE_DLSS5,
  MODE_GPU,
  MODE_LOOK,
  MODE_OPTIONS,
  PRESETS,
  PresetKey,
  UPSCALE_SLIDERS,
  UPSCALER_CATMULL,
  UPSCALER_GSR,
  UPSCALER_OPTIONS,
} from './core/lookSettings';
import { estimateSd8Gen3Budget } from './core/sd8Gen3Profile';
import { NeuralModelInfo, parseNeuralModelFile } from './core/tfliteModel';
import { PipelineTelemetry } from './gl/GlPipeline';
import { ViewportCanvas } from './components/ViewportCanvas';

const MODEL_STORAGE_KEY = 'gamelens_neural_model_meta';

function loadSavedModelMeta(): NeuralModelInfo | null {
  try {
    const raw = localStorage.getItem(MODEL_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as NeuralModelInfo) : null;
  } catch {
    return null;
  }
}

export function App() {
  const [settings, setSettings] = useState<LookState>(DEFAULT_LOOK_STATE);
  const [activePreset, setActivePreset] = useState<PresetKey | null>('CRISP');
  const [crashLog, setCrashLog] = useState<string>(() => readCrashLog());
  const [crashCopied, setCrashCopied] = useState(false);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [clipName, setClipName] = useState<string | null>(null);
  const [clipIsVideo, setClipIsVideo] = useState(false);

  const [neuralModel, setNeuralModel] = useState<NeuralModelInfo | null>(() => loadSavedModelMeta());
  const [modelError, setModelError] = useState<string | null>(null);

  const [use3dEngine, setUse3dEngine] = useState(false);

  const [telemetry, setTelemetry] = useState<PipelineTelemetry>({
    status: 'Neural idle',
    stats: '',
    diag: '',
    captureWidth: 0,
    captureHeight: 0,
    outputWidth: 0,
    outputHeight: 0,
    lutLoaded: false,
    lutSize: 33,
    gsrAvailable: true,
    dlss5Available: false,
    dlss5Tier: 1,
    dlss5GbufMs: 0,
    dlss5ReconMs: 0,
    dlss5FrameGen: false,
  });

  const canDisplayCapture =
    typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices?.getDisplayMedia === 'function';

  const modelInputRef = useRef<HTMLInputElement | null>(null);
  const clipInputRef = useRef<HTMLInputElement | null>(null);

  // Global uncaught error handler (mirrors GameLensApp.kt)
  useEffect(() => {
    const onErr = (ev: ErrorEvent) => {
      setCrashLog(recordCrash('uncaught error', ev.error ?? ev.message));
    };
    const onRej = (ev: PromiseRejectionEvent) => {
      setCrashLog(recordCrash('unhandled rejection', ev.reason));
    };
    window.addEventListener('error', onErr);
    window.addEventListener('unhandledrejection', onRej);
    return () => {
      window.removeEventListener('error', onErr);
      window.removeEventListener('unhandledrejection', onRej);
    };
  }, []);

  // Cleanup object URLs on unmount/replace
  useEffect(() => {
    return () => {
      if (clipUrl) URL.revokeObjectURL(clipUrl);
    };
  }, [clipUrl]);

  const stopCapture = useCallback(() => {
    setStream((prev) => {
      if (prev) prev.getTracks().forEach((t) => t.stop());
      return null;
    });
  }, []);

  const startCapture = useCallback(async () => {
    setCaptureError(null);
    if (!navigator.mediaDevices?.getDisplayMedia) {
      setCaptureError(
        'Screen sharing is unavailable in mobile browsers. Use Load clip to test your own gameplay video or screenshot.',
      );
      return;
    }
    try {
      const media = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 120 } },
        audio: false,
      });
      const [track] = media.getVideoTracks();
      if (track) {
        track.addEventListener('ended', () => {
          setStream(null);
        });
      }
      setStream((prev) => {
        if (prev) prev.getTracks().forEach((t) => t.stop());
        return media;
      });
      setUse3dEngine(false);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'NotAllowedError') {
        setCaptureError('Screen share permission was cancelled.');
      } else {
        const msg = err instanceof Error ? err.message : String(err);
        setCaptureError(`Could not start screen capture: ${msg}`);
        setCrashLog(recordCrash('startCapture', err));
      }
    }
  }, []);

  const handleModelFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setModelError(null);
    try {
      const buf = new Uint8Array(await file.arrayBuffer());
      const info = parseNeuralModelFile(file.name, buf);
      setNeuralModel(info);
      try {
        localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(info));
      } catch {
        // ignore storage quota errors
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setModelError(`Could not load model: ${msg}`);
    }
  };

  const handleClipFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    stopCapture();
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    const url = URL.createObjectURL(file);
    setClipUrl(url);
    setClipName(file.name);
    setClipIsVideo(file.type.startsWith('video/'));
    setUse3dEngine(false);
  };

  const resetToSample = () => {
    stopCapture();
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    setClipUrl(null);
    setClipName(null);
    setClipIsVideo(false);
    setUse3dEngine(false);
  };

  const switchTo3dEngine = () => {
    stopCapture();
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    setClipUrl(null);
    setClipName(null);
    setClipIsVideo(false);
    setUse3dEngine(true);
    setSettings((prev) => (prev.mode === MODE_DLSS5 ? prev : { ...prev, mode: MODE_DLSS5 }));
  };

  const updateSlider = (key: keyof LookState, value: number) => {
    setActivePreset(null);
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handlePreset = (key: PresetKey) => {
    setActivePreset(key);
    setSettings((prev) => applyPreset(prev, key));
  };

  const applySd8Gen3Codm120 = () => {
    setActivePreset('SD8_GEN3_HYPER');
    setSettings((prev) => ({
      ...applyPreset(prev, 'SD8_GEN3_HYPER'),
      mode: MODE_GPU,
      upscaler: UPSCALER_GSR,
    }));
  };

  const sd8Budget = estimateSd8Gen3Budget(
    settings,
    telemetry.outputWidth || 2400,
    telemetry.outputHeight || 1080,
  );

  const handleCopyAndClearCrash = async () => {
    try {
      await navigator.clipboard.writeText(crashLog);
    } catch {
      // clipboard API may be restricted in non-HTTPS iframe
    }
    clearCrashLog();
    setCrashLog('');
    setCrashCopied(true);
    setTimeout(() => setCrashCopied(false), 3000);
  };

  const modeSummary =
    settings.bypass
      ? 'Bypass (raw capture)'
      : settings.mode === MODE_DLSS5
      ? use3dEngine && !stream && !clipUrl
        ? `DLSS-5 Tier ${telemetry.dlss5Tier}${telemetry.dlss5FrameGen ? ' · FG 2x (120Hz)' : ''}`
        : 'DLSS-5 Spatial + HyperReal LUT'
      : settings.mode === MODE_LOOK
      ? 'Look filters only'
      : settings.mode === MODE_GPU
      ? settings.upscaler === UPSCALER_GSR && telemetry.gsrAvailable
        ? 'Qualcomm GSR + HyperReal LUT'
        : 'Catmull-Rom + HyperReal LUT'
      : 'Auto / Neural fallback';

  const sourceLabel = stream
    ? 'Live screen capture'
    : clipName
    ? clipName
    : use3dEngine
    ? '3D G-Buffer Engine (drag to orbit)'
    : 'Built-in gameplay sample';

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100">
      {/* Top Bar: compact <= 52px on mobile, strict 3-zone contract */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0B0F17]/95 backdrop-blur-sm">
        <div className="mx-auto flex h-[52px] max-w-[1440px] items-center justify-between gap-8 px-4 sm:px-6">
          <a
            href="#viewport"
            className="font-display text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
          >
            GameLens
          </a>

          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-400">
            <a
              href="#viewport"
              className="hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              Viewport
            </a>
            <a
              href="#neural-model"
              className="hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              Neural Model
            </a>
            <a
              href="#modes"
              className="hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              Modes
            </a>
            <a
              href="#look-sliders"
              className="hover:text-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              Look Sliders
            </a>
          </nav>

          <div className="flex items-center gap-3 shrink-0">
            {canDisplayCapture ? (
              stream ? (
                <button
                  type="button"
                  onClick={stopCapture}
                  className="inline-flex min-h-[40px] items-center gap-2 rounded-lg bg-rose-600 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-rose-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-400 whitespace-nowrap shrink-0"
                >
                  <Square className="h-3.5 w-3.5" />
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startCapture}
                  className="inline-flex min-h-[40px] items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-sky-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300 whitespace-nowrap shrink-0"
                >
                  <Play className="h-3.5 w-3.5" />
                  Start
                </button>
              )
            ) : (
              <button
                type="button"
                onClick={() => clipInputRef.current?.click()}
                className="inline-flex min-h-[40px] items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-sky-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-300 whitespace-nowrap shrink-0"
              >
                <Upload className="h-3.5 w-3.5" />
                Load clip
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6 sm:py-6">
        {/* Crash Log Banner (matches MainActivity.kt crashFile block) */}
        {crashLog && (
          <section className="mb-6 rounded-xl border border-rose-900/70 bg-rose-950/30 p-4">
            <p className="text-sm font-semibold text-rose-300">
              GameLens recorded a problem last time:
            </p>
            <pre className="mt-2 max-h-40 overflow-auto rounded-lg border border-rose-900/50 bg-[#0B0F17] p-3 font-mono text-xs text-rose-300 whitespace-pre-wrap">
              {crashLog.slice(-1800)}
            </pre>
            <button
              type="button"
              onClick={handleCopyAndClearCrash}
              className="mt-3 inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-rose-700/60 bg-rose-900/40 px-4 py-2.5 text-xs font-medium text-rose-100 transition-colors hover:bg-rose-900/70 whitespace-nowrap shrink-0"
            >
              <Copy className="h-3.5 w-3.5" />
              Copy crash log, then clear it
            </button>
          </section>
        )}

        {crashCopied && (
          <p className="mb-4 text-xs text-slate-300">Crash log copied and cleared.</p>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 lg:gap-8 lg:items-start">
          {/* Left Column: Live WebGL2 Preview + Source & Presets */}
          <div id="viewport" className="space-y-5 lg:col-span-7 xl:col-span-8">
            <div className="space-y-1">
              <h1 className="font-display text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
                Real-time Gameplay Post-Process &amp; Upscaler
              </h1>
              <p className="text-xs text-slate-400">
                <span>Build 0.5</span>
                <span aria-hidden="true" className="mx-2">·</span>
                <span>Crisp default, HDR look, crash log</span>
                <span aria-hidden="true" className="mx-2">·</span>
                <span>
                  {telemetry.lutLoaded
                    ? `HyperReal Rec709 LUT (${telemetry.lutSize}³)`
                    : 'HyperReal LUT loading'}
                </span>
              </p>
              <p className="pt-1 text-sm leading-relaxed text-slate-300">
                Real-time sharpening, grading and glow for your game.{' '}
                {canDisplayCapture
                  ? 'When prompted what to share, choose a single game window or tab to avoid capture feedback.'
                  : 'Test live on the built-in gameplay frame or load a video/screenshot clip from your phone.'}
              </p>
            </div>

            <ViewportCanvas
              settings={settings}
              neuralModel={neuralModel}
              stream={stream}
              clipUrl={clipUrl}
              clipIsVideo={clipIsVideo}
              use3dEngine={use3dEngine}
              onTelemetry={setTelemetry}
              onCrash={setCrashLog}
            />

            {/* Quick Section Jump Bar on Mobile */}
            <nav
              aria-label="Mobile section jump"
              className="grid grid-cols-3 gap-2 md:hidden"
            >
              <a
                href="#modes"
                className="flex min-h-[44px] items-center justify-center rounded-lg border border-slate-800 bg-slate-900/60 px-3 text-xs font-medium text-slate-200 whitespace-nowrap"
              >
                Modes
              </a>
              <a
                href="#look-sliders"
                className="flex min-h-[44px] items-center justify-center rounded-lg border border-slate-800 bg-slate-900/60 px-3 text-xs font-medium text-slate-200 whitespace-nowrap"
              >
                Look Sliders
              </a>
              <a
                href="#neural-model"
                className="flex min-h-[44px] items-center justify-center rounded-lg border border-slate-800 bg-slate-900/60 px-3 text-xs font-medium text-slate-200 whitespace-nowrap"
              >
                Neural Model
              </a>
            </nav>

            {/* Source & Capture Toolbar */}
            <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3.5 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
              <div className="min-w-0 space-y-0.5">
                <p className="truncate text-sm font-medium text-slate-200">{sourceLabel}</p>
                <p className="font-mono text-xs text-slate-400">
                  <span>{modeSummary}</span>
                  {telemetry.captureWidth > 0 && (
                    <>
                      <span aria-hidden="true" className="mx-2">·</span>
                      <span>
                        {telemetry.captureWidth}×{telemetry.captureHeight} → {telemetry.outputWidth}×
                        {telemetry.outputHeight}
                      </span>
                    </>
                  )}
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2 shrink-0">
                {canDisplayCapture && (
                  <button
                    type="button"
                    onClick={stream ? stopCapture : startCapture}
                    className={`inline-flex min-h-[44px] items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                      stream
                        ? 'bg-rose-600 text-white hover:bg-rose-500'
                        : 'bg-sky-500 text-slate-950 hover:bg-sky-400'
                    }`}
                  >
                    <Monitor className="h-3.5 w-3.5" />
                    {stream ? 'Stop capture' : 'Share game window'}
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => clipInputRef.current?.click()}
                  className={`inline-flex min-h-[44px] items-center gap-2 rounded-lg px-4 py-2.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    canDisplayCapture
                      ? 'border border-slate-700 bg-slate-800/80 text-slate-200 hover:bg-slate-800'
                      : 'bg-sky-500 text-slate-950 hover:bg-sky-400'
                  }`}
                >
                  <Upload className="h-3.5 w-3.5" />
                  Load clip
                </button>
                <input
                  ref={clipInputRef}
                  type="file"
                  accept="video/*,image/*"
                  onChange={handleClipFile}
                  className="hidden"
                />

                <button
                  type="button"
                  onClick={use3dEngine && !stream && !clipUrl ? resetToSample : switchTo3dEngine}
                  disabled={!telemetry.dlss5Available && !use3dEngine}
                  className={`inline-flex min-h-[44px] items-center gap-2 rounded-lg border px-4 py-2.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                    use3dEngine && !stream && !clipUrl
                      ? 'border-sky-400 bg-sky-500/15 text-sky-200'
                      : 'border-slate-700 bg-slate-800/80 text-slate-200 hover:bg-slate-800 disabled:opacity-40'
                  }`}
                >
                  <Box className="h-3.5 w-3.5" />
                  {use3dEngine && !stream && !clipUrl ? 'Sample frame' : '3D Engine'}
                </button>

                {(stream || clipUrl) && (
                  <button
                    type="button"
                    onClick={resetToSample}
                    className="inline-flex min-h-[44px] items-center gap-1.5 rounded-lg border border-slate-700 px-3.5 py-2.5 text-xs font-medium text-slate-300 transition-colors hover:bg-slate-800 whitespace-nowrap shrink-0"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Sample frame
                  </button>
                )}
              </div>
            </div>

            {captureError && <p className="text-xs text-amber-400">{captureError}</p>}

            {/* Presets Grid */}
            <section className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="font-display text-base font-semibold text-slate-100">Presets</h2>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={applySd8Gen3Codm120}
                    className="inline-flex min-h-[36px] items-center gap-1.5 rounded-lg border border-sky-500/40 bg-sky-500/10 px-3 py-1.5 text-xs font-semibold text-sky-200 transition-colors hover:bg-sky-500/20 whitespace-nowrap"
                  >
                    <Zap className="h-3.5 w-3.5" />
                    SD 8 Gen 3 · CoD Mobile 120Hz
                  </button>
                  <span className="text-xs text-slate-400">
                    {activePreset ? PRESETS[activePreset].label : 'Custom'}
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-4">
                {(Object.keys(PRESETS) as PresetKey[]).map((key) => {
                  const preset = PRESETS[key];
                  const isActive = activePreset === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => handlePreset(key)}
                      className={`min-h-[44px] rounded-lg border px-3 py-2.5 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
                        isActive
                          ? 'border-sky-400 bg-sky-500/15 text-sky-200'
                          : 'border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700 hover:bg-slate-800/80'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Neural Model Section (exact MainActivity.kt addAiSection) */}
            <section
              id="neural-model"
              className="space-y-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4 sm:p-5"
            >
              <h2 className="font-display text-base font-semibold text-slate-100">
                Neural model (Snapdragon NPU)
              </h2>
              <p className="text-xs leading-relaxed text-slate-400">
                Pick the int8 planar 2x .tflite from AI Hub (.tflite or the .zip). Its input size
                must match the game&apos;s render size, for example 1168x540 for a game rendering at
                1170x540. Then choose Neural or Auto in Modes.
              </p>

              <div className="flex flex-wrap items-center gap-4">
                <button
                  type="button"
                  onClick={() => modelInputRef.current?.click()}
                  className="inline-flex min-h-[44px] items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2.5 text-xs font-medium text-slate-100 transition-colors hover:bg-slate-700 whitespace-nowrap shrink-0"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Pick neural model (.tflite / .zip)
                </button>
                <input
                  ref={modelInputRef}
                  type="file"
                  accept=".tflite,.zip"
                  onChange={handleModelFile}
                  className="hidden"
                />
                <span className="text-sm text-slate-300">
                  Model: {neuralModel ? neuralModel.name : '(none)'}
                </span>
              </div>

              {modelError && <p className="text-xs text-rose-400">{modelError}</p>}

              <div className="rounded-lg border border-slate-800/80 bg-[#0B0F17] p-3 font-mono text-xs text-slate-300 space-y-1">
                <div>{telemetry.status}</div>
                {telemetry.stats && <div className="text-slate-400">{telemetry.stats}</div>}
                {telemetry.diag && <div className="text-sky-300">{telemetry.diag}</div>}
              </div>
            </section>
          </div>

          {/* Right Column: Modes, GPU Upscaler & Live Look Sliders */}
          <aside className="space-y-6 lg:col-span-5 xl:col-span-4">
            {/* Mode & Capture Options */}
            <section
              id="modes"
              className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4 sm:p-5"
            >
              <h2 className="font-display text-base font-semibold text-slate-100">Mode</h2>
              <div className="space-y-2" role="radiogroup" aria-label="Processing Mode">
                {MODE_OPTIONS.map((opt) => (
                  <label
                    key={opt.id}
                    className={`flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg border px-3.5 py-2.5 text-xs transition-colors ${
                      settings.mode === opt.id
                        ? 'border-sky-500/60 bg-sky-500/10 text-slate-100'
                        : 'border-slate-800/90 bg-slate-950/50 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <input
                      type="radio"
                      name="gamelens-mode"
                      checked={settings.mode === opt.id}
                      onChange={() => {
                        setSettings((prev) => ({ ...prev, mode: opt.id }));
                        if (opt.id === MODE_DLSS5 && !stream && !clipUrl && telemetry.dlss5Available) {
                          setUse3dEngine(true);
                        }
                      }}
                      className="h-4 w-4 shrink-0 accent-sky-400"
                    />
                    <span className="leading-snug">{opt.label}</span>
                  </label>
                ))}
              </div>

              {settings.mode === MODE_DLSS5 && (
                <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-950/60 p-3.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-sky-300">DLSS-5 Neural &amp; 120Hz Pipeline</span>
                    <span className="font-mono text-slate-400 tabular-nums">
                      Tier {telemetry.dlss5Tier} · {telemetry.dlss5GbufMs.toFixed(1)}+{telemetry.dlss5ReconMs.toFixed(1)} ms
                    </span>
                  </div>

                  <div>
                    <div className="flex items-center justify-between text-xs">
                      <label htmlFor="slider-neuralLighting" className="text-slate-300">
                        Neural lighting (3D-guided AO &amp; specular)
                      </label>
                      <span className="font-mono text-slate-200 tabular-nums">
                        {Number(settings.neuralLighting ?? DEFAULT_LOOK_STATE.neuralLighting).toFixed(2)}
                      </span>
                    </div>
                    <input
                      id="slider-neuralLighting"
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={Number(settings.neuralLighting ?? DEFAULT_LOOK_STATE.neuralLighting)}
                      onChange={(e) => updateSlider('neuralLighting', Number.parseFloat(e.target.value))}
                      className="gamelens-range"
                    />
                  </div>

                  <label className="flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg px-1 text-xs text-slate-200">
                    <input
                      type="checkbox"
                      checked={settings.frameGen}
                      onChange={(e) =>
                        setSettings((prev) => ({ ...prev, frameGen: e.target.checked }))
                      }
                      className="h-4 w-4 shrink-0 rounded accent-sky-400"
                    />
                    <span>Frame Generation (2x midpoint warp · 120Hz)</span>
                  </label>

                  {!use3dEngine && (stream || clipUrl) && (
                    <p className="text-xs text-slate-400">
                      Switch source to 3D Engine to inspect G-buffer motion vectors, Halton TAAU, and Frame Gen.
                    </p>
                  )}
                </div>
              )}

              <div className="rounded-lg border border-slate-800 bg-[#0B0F17] p-3 space-y-1">
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-semibold text-slate-200">Snapdragon 8 Gen 3 · Adreno 750</span>
                  <span
                    className={`font-mono tabular-nums ${
                      sd8Budget.verdict === 'LOCKED_120'
                        ? 'text-emerald-300'
                        : sd8Budget.verdict === 'TIGHT_120'
                        ? 'text-amber-300'
                        : 'text-rose-300'
                    }`}
                  >
                    ~{sd8Budget.adreno750Ms.toFixed(2)} / {sd8Budget.deadlineMs} ms · {sd8Budget.projectedFps}fps
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">{sd8Budget.gameRecommendation}</p>
              </div>

              <label className="flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg px-1 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={settings.captureHalf}
                  onChange={(e) =>
                    setSettings((prev) => ({ ...prev, captureHalf: e.target.checked }))
                  }
                  className="h-4 w-4 shrink-0 rounded accent-sky-400"
                />
                <span>Capture at half screen size (applies on next start)</span>
              </label>

              <div className="border-t border-slate-800/80 pt-4 space-y-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">GPU upscaler</h3>
                <div className="space-y-2" role="radiogroup" aria-label="GPU upscaler">
                  {UPSCALER_OPTIONS.map((opt) => (
                    <label
                      key={opt.id}
                      className={`flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg border px-3.5 py-2.5 text-xs transition-colors ${
                        settings.upscaler === opt.id
                          ? 'border-sky-500/60 bg-sky-500/10 text-slate-100'
                          : 'border-slate-800/90 bg-slate-950/50 text-slate-300 hover:border-slate-700'
                      }`}
                    >
                      <input
                        type="radio"
                        name="gamelens-upscaler"
                        checked={settings.upscaler === opt.id}
                        onChange={() => setSettings((prev) => ({ ...prev, upscaler: opt.id }))}
                        className="h-4 w-4 shrink-0 accent-sky-400"
                      />
                      <span className="leading-snug">{opt.label}</span>
                    </label>
                  ))}
                </div>

                <div className="space-y-2.5 pt-2">
                  {UPSCALE_SLIDERS.map((spec) => {
                    const rawVal = settings[spec.key] ?? DEFAULT_LOOK_STATE[spec.key];
                    const val = typeof rawVal === 'number' && Number.isFinite(rawVal) ? rawVal : 0;
                    const gsrOnly = spec.key !== 'upSharp';
                    const dlssActive = settings.mode === MODE_DLSS5 && use3dEngine && !stream && !clipUrl;
                    const dimmed = gsrOnly && (settings.upscaler === UPSCALER_CATMULL || dlssActive);
                    return (
                      <div key={spec.key} className={dimmed ? 'opacity-50' : ''}>
                        <div className="flex items-center justify-between text-xs">
                          <label htmlFor={`slider-${spec.key}`} className="text-slate-300">
                            {spec.label}
                          </label>
                          <span className="font-mono text-slate-200">{val.toFixed(2)}</span>
                        </div>
                        <input
                          id={`slider-${spec.key}`}
                          type="range"
                          min={spec.min}
                          max={spec.max}
                          step={(spec.max - spec.min) / 100}
                          value={val}
                          onChange={(e) => updateSlider(spec.key, Number.parseFloat(e.target.value))}
                          className="gamelens-range"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            {/* Look Sliders & Bypass */}
            <section
              id="look-sliders"
              className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4 sm:p-5"
            >
              <div className="flex items-center justify-between gap-4">
                <h2 className="font-display text-base font-semibold text-slate-100">
                  Look controls
                </h2>
                <span className="text-xs text-slate-400">
                  {settings.mode === MODE_LOOK
                    ? 'Full Look pass'
                    : 'Fused into GSR / DLSS-5 at 120Hz'}
                </span>
              </div>

              <label className="flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg border border-slate-800 bg-slate-950/60 px-3.5 py-2.5 text-xs font-medium text-slate-200">
                <input
                  type="checkbox"
                  checked={settings.bypass}
                  onChange={(e) => setSettings((prev) => ({ ...prev, bypass: e.target.checked }))}
                  className="h-4 w-4 shrink-0 rounded accent-sky-400"
                />
                <span>Bypass look (raw capture-rate test)</span>
              </label>

              <div className="space-y-2.5">
                {LOOK_SLIDERS.map((spec) => {
                  const rawVal = settings[spec.key] ?? DEFAULT_LOOK_STATE[spec.key];
                  const val = typeof rawVal === 'number' && Number.isFinite(rawVal) ? rawVal : 0;
                  const lookOnlyKeys = new Set<keyof LookState>([
                    'sharpen',
                    'clarity',
                    'vibrance',
                    'chroma',
                    'motion',
                  ]);
                  const lookOnly = lookOnlyKeys.has(spec.key);
                  const dimmed = settings.bypass || (lookOnly && settings.mode !== MODE_LOOK);
                  return (
                    <div key={spec.key} className={dimmed ? 'opacity-55' : ''}>
                      <div className="flex items-center justify-between text-xs">
                        <label htmlFor={`slider-${spec.key}`} className="text-slate-300">
                          {spec.label}
                        </label>
                        <span className="font-mono text-slate-200">{val.toFixed(2)}</span>
                      </div>
                      <input
                        id={`slider-${spec.key}`}
                        type="range"
                        min={spec.min}
                        max={spec.max}
                        step={(spec.max - spec.min) / 100}
                        value={val}
                        onChange={(e) => updateSlider(spec.key, Number.parseFloat(e.target.value))}
                        className="gamelens-range"
                      />
                    </div>
                  );
                })}
              </div>
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;
