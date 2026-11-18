import React, { useEffect, useRef, useState } from 'react';
import { Monitor, Upload, ZoomIn, Columns2, Image as ImageIcon, Play, Pause } from 'lucide-react';
import { parseCube } from '../engine/cubeLut';
import { WebGlPipeline, type PipelineDiagnostics } from '../engine/glPipeline';
import type { LookSettingsState } from '../engine/settings';
import type { NeuralModelInfo } from '../engine/tfliteParser';

import tacticalArenaUrl from '../assets/images/sample_tactical_arena_1795023762945.jpg';
import alpineExpeditionUrl from '../assets/images/sample_alpine_expedition_1795023762912.jpg';

export type SourceKind = 'tactical' | 'alpine' | 'upload' | 'capture';

interface ViewportCanvasProps {
  settings: LookSettingsState;
  neuralModel: NeuralModelInfo | null;
  sourceKind: SourceKind;
  onSelectSourceKind: (kind: SourceKind) => void;
  captureStream: MediaStream | null;
  onStartCapture: () => void;
  onStopCapture: () => void;
  onDiagnostics: (diag: PipelineDiagnostics) => void;
  onRecordError: (where: string, err: unknown) => void;
}

export const ViewportCanvas: React.FC<ViewportCanvasProps> = ({
  settings,
  neuralModel,
  sourceKind,
  onSelectSourceKind,
  captureStream,
  onStartCapture,
  onStopCapture,
  onDiagnostics,
  onRecordError,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pipelineRef = useRef<WebGlPipeline | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const neuralModelRef = useRef(neuralModel);
  neuralModelRef.current = neuralModel;

  const [splitEnabled, setSplitEnabled] = useState(true);
  const [splitX, setSplitX] = useState(0.5);
  const [zoom2x, setZoom2x] = useState(false);
  const [zoomCenter, setZoomCenter] = useState<[number, number]>([0.5, 0.5]);
  const [isDraggingSplit, setIsDraggingSplit] = useState(false);
  const [uploadedMedia, setUploadedMedia] = useState<{
    type: 'image' | 'video';
    url: string;
    name: string;
  } | null>(null);
  const [videoPaused, setVideoPaused] = useState(false);
  const [sourceReady, setSourceReady] = useState(false);
  const [lutError, setLutError] = useState<string | null>(null);
  const [webglError, setWebglError] = useState<string | null>(null);
  const [localDiag, setLocalDiag] = useState<PipelineDiagnostics | null>(null);

  const splitRef = useRef({ enabled: splitEnabled, x: splitX, zoom: zoom2x ? 2.0 : 1.0, zoomCenter });
  splitRef.current = { enabled: splitEnabled, x: splitX, zoom: zoom2x ? 2.0 : 1.0, zoomCenter };

  // Initialize WebGL2 pipeline and load HyperReal 3D LUT once
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;

    try {
      const pipeline = new WebGlPipeline(canvas);
      pipelineRef.current = pipeline;

      fetch('/assets/hyperreal_rec709_33.cube')
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.text();
        })
        .then((text) => {
          if (cancelled) return;
          const lut = parseCube(text);
          pipeline.uploadLut(lut);
        })
        .catch((err) => {
          if (cancelled) return;
          const msg = err instanceof Error ? err.message : String(err);
          setLutError(`LUT unavailable (${msg}), colour grade step skipped`);
          onRecordError('loadLut', err);
        });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setWebglError(msg);
      onRecordError('initWebGL2', err);
    }

    return () => {
      cancelled = true;
      pipelineRef.current?.release();
      pipelineRef.current = null;
    };
  }, [onRecordError]);

  // Load current image/video/capture source
  useEffect(() => {
    setSourceReady(false);
    setVideoPaused(false);

    if (sourceKind === 'capture') {
      const video = videoRef.current;
      if (video && captureStream) {
        video.srcObject = captureStream;
        video.muted = true;
        video
          .play()
          .then(() => setSourceReady(true))
          .catch((err) => onRecordError('capturePlay', err));
      }
      return;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject = null;
    }

    if (sourceKind === 'upload' && uploadedMedia) {
      if (uploadedMedia.type === 'video') {
        const video = videoRef.current;
        if (video) {
          video.srcObject = null;
          video.src = uploadedMedia.url;
          video.loop = true;
          video.muted = true;
          video
            .play()
            .then(() => setSourceReady(true))
            .catch((err) => onRecordError('videoPlay', err));
        }
        return;
      } else {
        const img = new Image();
        img.onload = () => {
          imageRef.current = img;
          setSourceReady(true);
        };
        img.onerror = () => onRecordError('imageLoad', new Error('Failed to load uploaded image'));
        img.src = uploadedMedia.url;
        return;
      }
    }

    const sampleUrl = sourceKind === 'alpine' ? alpineExpeditionUrl : tacticalArenaUrl;
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setSourceReady(true);
    };
    img.onerror = () => onRecordError('sampleLoad', new Error(`Failed to load ${sampleUrl}`));
    img.src = sampleUrl;
  }, [sourceKind, captureStream, uploadedMedia, onRecordError]);

  // Render loop
  useEffect(() => {
    let rafId = 0;
    let lastDiagReport = 0;

    const tick = (now: number) => {
      const canvas = canvasRef.current;
      const pipeline = pipelineRef.current;
      if (canvas && pipeline && sourceReady) {
        const useVideo =
          sourceKind === 'capture' || (sourceKind === 'upload' && uploadedMedia?.type === 'video');
        const el = useVideo ? videoRef.current : imageRef.current;
        const srcW = useVideo
          ? (videoRef.current?.videoWidth ?? 0)
          : (imageRef.current?.naturalWidth ?? 0);
        const srcH = useVideo
          ? (videoRef.current?.videoHeight ?? 0)
          : (imageRef.current?.naturalHeight ?? 0);

        if (el && srcW > 0 && srcH > 0) {
          // Match canvas backing store to source aspect at crisp viewport resolution (capped at 1920px wide)
          const targetW = Math.min(srcW, 1920);
          const targetH = Math.max(1, Math.round((targetW * srcH) / srcW));
          if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
          }

          try {
            const diag = pipeline.renderFrame(
              el,
              srcW,
              srcH,
              canvas.width,
              canvas.height,
              settingsRef.current,
              neuralModelRef.current,
              splitRef.current,
            );
            if (now - lastDiagReport > 250) {
              lastDiagReport = now;
              setLocalDiag(diag);
              onDiagnostics(diag);
            }
          } catch (err) {
            onRecordError('renderFrame', err);
          }
        }
      }
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [sourceReady, sourceKind, uploadedMedia, onDiagnostics, onRecordError]);

  const handleFilePick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (uploadedMedia) {
      URL.revokeObjectURL(uploadedMedia.url);
    }
    const url = URL.createObjectURL(file);
    const isVideo = file.type.startsWith('video/');
    setUploadedMedia({ type: isVideo ? 'video' : 'image', url, name: file.name });
    onSelectSourceKind('upload');
  };

  const updatePointerInteraction = (clientX: number, clientY: number, rect: DOMRect) => {
    const nx = Math.min(Math.max((clientX - rect.left) / rect.width, 0.05), 0.95);
    const ny = Math.min(Math.max(1 - (clientY - rect.top) / rect.height, 0.05), 0.95);
    if (isDraggingSplit) {
      setSplitX(nx);
    } else if (zoom2x) {
      setZoomCenter([nx, ny]);
    }
  };

  const togglePlayPause = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play();
      setVideoPaused(false);
    } else {
      v.pause();
      setVideoPaused(true);
    }
  };

  const isVideoActive =
    sourceKind === 'capture' || (sourceKind === 'upload' && uploadedMedia?.type === 'video');

  return (
    <div className="flex flex-col gap-4">
      {/* Hidden media element used for video files & MediaStream screen capture */}
      <video ref={videoRef} playsInline className="hidden" />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,video/*"
        onChange={handleFilePick}
        className="hidden"
      />

      {/* Source & Comparison Toolbar (Workspace Toolbar, below header) */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#232C3D] bg-[#121721] px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => onSelectSourceKind('tactical')}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              sourceKind === 'tactical'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-[#1B2333] hover:text-white'
            }`}
          >
            <ImageIcon className="h-3.5 w-3.5" />
            Tactical Arena
          </button>
          <button
            type="button"
            onClick={() => onSelectSourceKind('alpine')}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              sourceKind === 'alpine'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-[#1B2333] hover:text-white'
            }`}
          >
            <ImageIcon className="h-3.5 w-3.5" />
            Alpine RPG
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              sourceKind === 'upload'
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-[#1B2333] hover:text-white'
            }`}
          >
            <Upload className="h-3.5 w-3.5" />
            {uploadedMedia ? uploadedMedia.name.slice(0, 18) : 'Load Clip / Image'}
          </button>
          <button
            type="button"
            onClick={sourceKind === 'capture' && captureStream ? onStopCapture : onStartCapture}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              sourceKind === 'capture'
                ? 'bg-emerald-600 text-white hover:bg-emerald-500'
                : 'text-slate-300 hover:bg-[#1B2333] hover:text-white'
            }`}
          >
            <Monitor className="h-3.5 w-3.5" />
            {sourceKind === 'capture' && captureStream ? 'Sharing Screen' : 'Capture App / Screen'}
          </button>
        </div>

        <div className="flex items-center gap-2">
          {isVideoActive && sourceKind === 'upload' && (
            <button
              type="button"
              onClick={togglePlayPause}
              className="inline-flex items-center gap-1.5 rounded-md border border-[#2A354A] bg-[#181F2C] px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-[#212A3B] whitespace-nowrap shrink-0"
            >
              {videoPaused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
              {videoPaused ? 'Resume' : 'Pause'}
            </button>
          )}
          <button
            type="button"
            onClick={() => setSplitEnabled((v) => !v)}
            className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              splitEnabled
                ? 'border-blue-500/60 bg-blue-600/15 text-blue-300'
                : 'border-[#2A354A] bg-[#181F2C] text-slate-300 hover:text-white'
            }`}
          >
            <Columns2 className="h-3.5 w-3.5" />
            Before / After Wipe
          </button>
          <button
            type="button"
            onClick={() => setZoom2x((v) => !v)}
            className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
              zoom2x
                ? 'border-blue-500/60 bg-blue-600/15 text-blue-300'
                : 'border-[#2A354A] bg-[#181F2C] text-slate-300 hover:text-white'
            }`}
          >
            <ZoomIn className="h-3.5 w-3.5" />
            2x Inspect
          </button>
        </div>
      </div>

      {/* Main WebGL2 Viewport Frame */}
      <div className="relative overflow-hidden rounded-lg border border-[#232C3D] bg-[#07090D]">
        {webglError ? (
          <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 p-8 text-center">
            <p className="text-sm font-medium text-red-400">WebGL2 context unavailable</p>
            <p className="max-w-md text-xs text-slate-400">{webglError}</p>
          </div>
        ) : (
          <div
            className="relative aspect-video w-full select-none"
            onPointerDown={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = (e.clientX - rect.left) / rect.width;
              if (splitEnabled && Math.abs(relX - splitX) < 0.04) {
                setIsDraggingSplit(true);
                e.currentTarget.setPointerCapture(e.pointerId);
              } else if (zoom2x) {
                const nx = Math.min(Math.max(relX, 0.05), 0.95);
                const ny = Math.min(Math.max(1 - (e.clientY - rect.top) / rect.height, 0.05), 0.95);
                setZoomCenter([nx, ny]);
              }
            }}
            onPointerMove={(e) => {
              if (!isDraggingSplit && !(zoom2x && e.buttons === 1)) return;
              updatePointerInteraction(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect());
            }}
            onPointerUp={(e) => {
              if (isDraggingSplit) {
                setIsDraggingSplit(false);
                e.currentTarget.releasePointerCapture(e.pointerId);
              }
            }}
          >
            <canvas
              ref={canvasRef}
              className="h-full w-full object-contain"
              aria-label="GameLens WebGL2 output viewport"
            />

            {/* Split-screen divider handle */}
            {splitEnabled && (
              <div
                className="pointer-events-none absolute inset-y-0 w-px bg-white/80 shadow-[0_0_8px_rgba(0,0,0,0.8)]"
                style={{ left: `${splitX * 100}%` }}
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-md border border-white/30 bg-[#0B0E14]/90 px-2 py-1 text-[11px] font-medium text-white shadow">
                  Drag
                </div>
              </div>
            )}

            {/* Scrim-backed corner comparison labels */}
            {splitEnabled && (
              <>
                <div className="pointer-events-none absolute top-3 left-3 rounded bg-black/70 px-2.5 py-1 text-xs font-medium text-slate-200">
                  Original Input
                </div>
                <div className="pointer-events-none absolute top-3 right-3 rounded bg-black/70 px-2.5 py-1 text-xs font-medium text-blue-300">
                  {settings.bypass ? 'Bypass (Raw)' : 'GameLens Output'}
                </div>
              </>
            )}

            {!sourceReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-[#0B0E14]/80 text-xs text-slate-400">
                Loading source texture into WebGL2...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Measured Pipeline Metrics Strip (4-column KPI strip below viewport) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-[#232C3D] bg-[#121721] px-4 py-3">
          <p className="text-xs text-slate-400">Capture Surface</p>
          <p className="mt-1 font-mono text-sm font-medium text-slate-100">
            {localDiag ? `${localDiag.captureW} × ${localDiag.captureH}` : '—'}
          </p>
          <p className="mt-0.5 text-[11px] text-slate-500">
            {settings.captureHalf ? 'Half-res capture (2x target)' : 'Native capture'}
          </p>
        </div>

        <div className="rounded-lg border border-[#232C3D] bg-[#121721] px-4 py-3">
          <p className="text-xs text-slate-400">Presented Output</p>
          <p className="mt-1 font-mono text-sm font-medium text-slate-100">
            {localDiag ? `${localDiag.outputW} × ${localDiag.outputH}` : '—'}
          </p>
          <p className="mt-0.5 truncate text-[11px] text-slate-500">
            {localDiag?.activePath ?? 'Initializing'}
          </p>
        </div>

        <div className="rounded-lg border border-[#232C3D] bg-[#121721] px-4 py-3">
          <p className="text-xs text-slate-400">Render Cadence</p>
          <p className="mt-1 font-mono text-sm font-medium text-slate-100">
            {localDiag ? `${localDiag.drawnFps.toFixed(0)} fps` : '—'}
          </p>
          <p className="mt-0.5 font-mono text-[11px] text-slate-500">
            {localDiag ? `${localDiag.frameMs.toFixed(2)} ms / frame` : 'Measuring...'}
          </p>
        </div>

        <div className="rounded-lg border border-[#232C3D] bg-[#121721] px-4 py-3">
          <p className="text-xs text-slate-400">HyperReal 3D LUT</p>
          <p className="mt-1 font-mono text-sm font-medium text-slate-100">
            {localDiag?.lutLoaded ? `${localDiag.lutSize}³ Rec.709` : lutError ? 'Disabled' : 'Loading...'}
          </p>
          <p className="mt-0.5 font-mono text-[11px] text-slate-500">
            Strength {(settings.lut * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {lutError && (
        <p className="text-xs text-amber-400">{lutError}</p>
      )}
    </div>
  );
};
