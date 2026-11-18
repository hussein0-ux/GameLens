import { useCallback, useEffect, useState } from 'react';
import { ControlsPanel } from './components/ControlsPanel';
import { NeuralPanel } from './components/NeuralPanel';
import { ArchitecturePanel } from './components/ArchitecturePanel';
import { ViewportCanvas, type SourceKind } from './components/ViewportCanvas';
import type { PipelineDiagnostics } from './engine/glPipeline';
import {
  DEFAULT_LOOK_SETTINGS,
  type LookPreset,
  type LookSettingsState,
} from './engine/settings';
import type { NeuralModelInfo } from './engine/tfliteParser';

type TabKey = 'studio' | 'neural' | 'architecture';

const CRASH_KEY = 'gamelens_crash_log';
const SETTINGS_KEY = 'gamelens_look_settings';

function loadSavedSettings(): LookSettingsState {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_LOOK_SETTINGS };
    return { ...DEFAULT_LOOK_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_LOOK_SETTINGS };
  }
}

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('studio');
  const [settings, setSettings] = useState<LookSettingsState>(loadSavedSettings);
  const [activePresetId, setActivePresetId] = useState<string | null>('crisp');
  const [sourceKind, setSourceKind] = useState<SourceKind>('tactical');
  const [captureStream, setCaptureStream] = useState<MediaStream | null>(null);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [neuralModel, setNeuralModel] = useState<NeuralModelInfo | null>(null);
  const [diagnostics, setDiagnostics] = useState<PipelineDiagnostics | null>(null);
  const [crashLog, setCrashLog] = useState<string>(() => {
    try {
      return localStorage.getItem(CRASH_KEY) ?? '';
    } catch {
      return '';
    }
  });

  const recordError = useCallback((where: string, err: unknown) => {
    const msg = err instanceof Error ? `${err.name}: ${err.message}\n${err.stack ?? ''}` : String(err);
    const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
    setCrashLog((prev) => {
      const next = `${prev}[${stamp}] ${where}\n${msg}\n\n`.slice(-20000);
      try {
        localStorage.setItem(CRASH_KEY, next);
      } catch {
        // ignore storage quota errors
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onErr = (e: ErrorEvent) => recordError('window.error', e.error ?? e.message);
    const onRej = (e: PromiseRejectionEvent) => recordError('unhandledrejection', e.reason);
    window.addEventListener('error', onErr);
    window.addEventListener('unhandledrejection', onRej);
    return () => {
      window.removeEventListener('error', onErr);
      window.removeEventListener('unhandledrejection', onRej);
    };
  }, [recordError]);

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      // ignore storage errors
    }
  }, [settings]);

  const handleUpdateSetting = useCallback(
    <K extends keyof LookSettingsState>(key: K, value: LookSettingsState[K]) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
      if (key !== 'mode' && key !== 'upscaler' && key !== 'captureHalf' && key !== 'bypass' && key !== 'lut') {
        setActivePresetId(null);
      }
    },
    [],
  );

  const handleApplyPreset = useCallback((preset: LookPreset) => {
    setSettings((prev) => ({
      ...prev,
      sharpen: preset.sharpen,
      clarity: preset.clarity,
      contrast: preset.contrast,
      saturation: preset.saturation,
      filmic: preset.filmic,
      exposure: preset.exposure,
      grade: preset.grade,
      vibrance: preset.vibrance,
      glow: preset.glow,
      vignette: preset.vignette,
      chroma: preset.chroma,
      motion: preset.motion,
      hdr: preset.hdr,
      blackPt: preset.blackPt,
    }));
    setActivePresetId(preset.id);
  }, []);

  const handleResetDefaults = useCallback(() => {
    setSettings({ ...DEFAULT_LOOK_SETTINGS });
    setActivePresetId('crisp');
  }, []);

  const stopCapture = useCallback(() => {
    setCaptureStream((prev) => {
      if (prev) {
        prev.getTracks().forEach((t) => t.stop());
      }
      return null;
    });
    setSourceKind((prev) => (prev === 'capture' ? 'tactical' : prev));
  }, []);

  const startCapture = useCallback(async () => {
    setCaptureNotice(null);
    if (!navigator.mediaDevices?.getDisplayMedia) {
      setCaptureNotice(
        'Screen capture (getDisplayMedia) is not supported in this browser context. Load a clip or image instead.',
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: { ideal: 120, max: 120 },
        },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      if (track) {
        track.onended = () => {
          setCaptureStream(null);
          setSourceKind((prev) => (prev === 'capture' ? 'tactical' : prev));
        };
      }
      setCaptureStream((prev) => {
        if (prev) prev.getTracks().forEach((t) => t.stop());
        return stream;
      });
      setSourceKind('capture');
      setActiveTab('studio');
    } catch (err) {
      const name = err instanceof Error ? err.name : '';
      if (name !== 'AbortError' && name !== 'NotAllowedError') {
        setCaptureNotice(err instanceof Error ? err.message : 'Unable to start screen capture');
      }
    }
  }, []);

  const copyAndClearCrash = useCallback(() => {
    if (crashLog && navigator.clipboard) {
      void navigator.clipboard.writeText(crashLog);
    }
    setCrashLog('');
    try {
      localStorage.removeItem(CRASH_KEY);
    } catch {
      // ignore
    }
  }, [crashLog]);

  return (
    <div className="min-h-screen bg-[#0B0E14] text-slate-100">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="border-b border-[#1E2636] bg-[#0B0E14]">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-8 px-6 py-4">
          {/* Zone 1: Single-line Brand Wordmark */}
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('studio');
            }}
            className="font-display text-lg font-bold tracking-tight text-white whitespace-nowrap shrink-0"
          >
            GameLens
          </a>

          {/* Zone 2: Clean single-line navigation links */}
          <nav className="flex items-center gap-6 text-sm font-medium text-slate-400">
            <button
              type="button"
              onClick={() => setActiveTab('studio')}
              className={`border-b-2 py-1 whitespace-nowrap shrink-0 transition-colors ${
                activeTab === 'studio'
                  ? 'border-blue-500 text-white'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Studio
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('neural')}
              className={`border-b-2 py-1 whitespace-nowrap shrink-0 transition-colors ${
                activeTab === 'neural'
                  ? 'border-blue-500 text-white'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Neural NPU
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('architecture')}
              className={`border-b-2 py-1 whitespace-nowrap shrink-0 transition-colors ${
                activeTab === 'architecture'
                  ? 'border-blue-500 text-white'
                  : 'border-transparent hover:text-slate-200'
              }`}
            >
              Architecture
            </button>
          </nav>

          {/* Zone 3: Single Primary Action */}
          <div className="flex items-center shrink-0">
            <button
              type="button"
              onClick={captureStream ? stopCapture : startCapture}
              className={`rounded-md px-4 py-2 text-xs font-semibold whitespace-nowrap shrink-0 transition-colors ${
                captureStream
                  ? 'bg-red-600 text-white hover:bg-red-500'
                  : 'bg-blue-600 text-white hover:bg-blue-500'
              }`}
            >
              {captureStream ? 'Stop Capture' : 'Start Capture'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] px-6 py-6">
        {captureNotice && (
          <div className="mb-5 flex items-center justify-between gap-4 rounded-md border border-amber-500/40 bg-amber-950/30 px-4 py-2.5 text-xs text-amber-200">
            <span>{captureNotice}</span>
            <button
              type="button"
              onClick={() => setCaptureNotice(null)}
              className="font-medium text-amber-300 hover:text-white whitespace-nowrap shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {crashLog && activeTab !== 'neural' && (
          <div className="mb-5 flex items-center justify-between gap-4 rounded-md border border-red-500/40 bg-red-950/30 px-4 py-2.5 text-xs text-red-200">
            <span>GameLens recorded a runtime error. View or clear it under Neural NPU.</span>
            <button
              type="button"
              onClick={() => setActiveTab('neural')}
              className="font-medium text-red-300 underline hover:text-white whitespace-nowrap shrink-0"
            >
              Open Diagnostics
            </button>
          </div>
        )}

        {activeTab === 'studio' && (
          <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-12">
            <div className="xl:col-span-8">
              <ViewportCanvas
                settings={settings}
                neuralModel={neuralModel}
                sourceKind={sourceKind}
                onSelectSourceKind={setSourceKind}
                captureStream={captureStream}
                onStartCapture={startCapture}
                onStopCapture={stopCapture}
                onDiagnostics={setDiagnostics}
                onRecordError={recordError}
              />
            </div>

            <div className="xl:col-span-4">
              <ControlsPanel
                settings={settings}
                activePresetId={activePresetId}
                onUpdateSetting={handleUpdateSetting}
                onApplyPreset={handleApplyPreset}
                onResetDefaults={handleResetDefaults}
              />
            </div>
          </div>
        )}

        {activeTab === 'neural' && (
          <NeuralPanel
            neuralModel={neuralModel}
            onModelLoaded={setNeuralModel}
            diagnostics={diagnostics}
            crashLog={crashLog}
            onCopyAndClearCrash={copyAndClearCrash}
          />
        )}

        {activeTab === 'architecture' && <ArchitecturePanel />}
      </main>
    </div>
  );
}

export default App;
