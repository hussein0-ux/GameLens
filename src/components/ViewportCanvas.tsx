import { PointerEvent, useEffect, useRef, useState } from 'react';
import { LookState } from '../core/lookSettings';
import { NeuralModelInfo } from '../core/tfliteModel';
import { recordCrash } from '../core/crashLog';
import { PipelineTelemetry, WebGlPipeline } from '../gl/GlPipeline';
import sampleFrameUrl from '../assets/images/gameplay_sample_frame_1795031451310.jpg';

export type SourceKind = 'sample' | 'engine3d' | 'capture' | 'clip';

interface ViewportCanvasProps {
  settings: LookState;
  neuralModel: NeuralModelInfo | null;
  stream: MediaStream | null;
  clipUrl: string | null;
  clipIsVideo: boolean;
  use3dEngine: boolean;
  onTelemetry: (telemetry: PipelineTelemetry) => void;
  onCrash: (log: string) => void;
}

export function ViewportCanvas({
  settings,
  neuralModel,
  stream,
  clipUrl,
  clipIsVideo,
  use3dEngine,
  onTelemetry,
  onCrash,
}: ViewportCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sampleImgRef = useRef<HTMLImageElement | null>(null);
  const clipImgRef = useRef<HTMLImageElement | null>(null);
  const pipelineRef = useRef<WebGlPipeline | null>(null);
  const dragRef = useRef<{ active: boolean; x: number; y: number }>({
    active: false,
    x: 0,
    y: 0,
  });

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const neuralRef = useRef(neuralModel);
  neuralRef.current = neuralModel;
  const streamRef = useRef(stream);
  streamRef.current = stream;
  const clipUrlRef = useRef(clipUrl);
  clipUrlRef.current = clipUrl;
  const clipIsVideoRef = useRef(clipIsVideo);
  clipIsVideoRef.current = clipIsVideo;
  const use3dEngineRef = useRef(use3dEngine);
  use3dEngineRef.current = use3dEngine;

  const [sampleReady, setSampleReady] = useState(false);
  const [sampleFailed, setSampleFailed] = useState(false);
  const [glError, setGlError] = useState<string | null>(null);

  // Initialize WebGL2 pipeline once
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let pipe: WebGlPipeline;
    try {
      pipe = new WebGlPipeline(canvas);
      pipelineRef.current = pipe;
      void pipe.loadCubeLut('/hyperreal_rec709_33.cube');
    } catch (err) {
      const log = recordCrash('WebGlPipeline.init', err);
      onCrash(log);
      setGlError(err instanceof Error ? err.message : String(err));
      return;
    }

    let rafId = 0;
    let lastTelemetryPush = 0;

    const renderLoop = (now: number) => {
      if (!pipelineRef.current) return;
      try {
        if (
          use3dEngineRef.current &&
          pipe.telemetry.dlss5Available &&
          !streamRef.current &&
          !clipUrlRef.current
        ) {
          if (canvas.width !== 1280 || canvas.height !== 720) {
            canvas.width = 1280;
            canvas.height = 720;
          }
          pipe.drawDlss5Frame(now / 1000, settingsRef.current);
        } else {
          const video = videoRef.current;
          const sImg = sampleImgRef.current;
          const cImg = clipImgRef.current;

          let activeEl: HTMLVideoElement | HTMLImageElement | null = null;
          let srcW = 0;
          let srcH = 0;

          if (
            (streamRef.current || (clipUrlRef.current && clipIsVideoRef.current)) &&
            video &&
            video.readyState >= 2 &&
            video.videoWidth > 0
          ) {
            activeEl = video;
            srcW = video.videoWidth;
            srcH = video.videoHeight;
          } else if (
            clipUrlRef.current &&
            !clipIsVideoRef.current &&
            cImg &&
            cImg.complete &&
            cImg.naturalWidth > 0
          ) {
            activeEl = cImg;
            srcW = cImg.naturalWidth;
            srcH = cImg.naturalHeight;
          } else if (sImg && sImg.complete && sImg.naturalWidth > 0) {
            activeEl = sImg;
            srcW = sImg.naturalWidth;
            srcH = sImg.naturalHeight;
          }

          if (activeEl && srcW > 0 && srcH > 0) {
            // Match canvas pixel resolution to source aspect ratio up to 1920px wide
            const targetW = Math.min(Math.max(srcW, 960), 1920);
            const targetH = Math.round((targetW * srcH) / srcW);
            if (canvas.width !== targetW || canvas.height !== targetH) {
              canvas.width = targetW;
              canvas.height = targetH;
            }
            pipe.drawFrame(activeEl, srcW, srcH, settingsRef.current, neuralRef.current);
          }
        }

        if (now - lastTelemetryPush >= 400) {
          lastTelemetryPush = now;
          onTelemetry({ ...pipe.telemetry });
        }
      } catch (err) {
        const log = recordCrash('drawFrame', err);
        onCrash(log);
      }
      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(rafId);
      pipe.release();
      pipelineRef.current = null;
    };
  }, [onCrash, onTelemetry]);

  // Attach MediaStream or video clip to hidden video element
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (stream) {
      video.srcObject = stream;
      video.removeAttribute('src');
      video.loop = false;
      void video.play().catch(() => {});
    } else if (clipUrl && clipIsVideo) {
      video.srcObject = null;
      video.src = clipUrl;
      video.loop = true;
      void video.play().catch(() => {});
    } else {
      video.pause();
      video.srcObject = null;
      video.removeAttribute('src');
    }
  }, [stream, clipUrl, clipIsVideo]);

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!use3dEngine || stream || clipUrl) return;
    dragRef.current = { active: true, x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current.active || !use3dEngine || stream || clipUrl) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current.x = e.clientX;
    dragRef.current.y = e.clientY;
    pipelineRef.current?.orbitDlss5Camera(dx, dy);
  };

  const handlePointerEnd = (e: PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current.active) {
      dragRef.current.active = false;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    }
  };

  const interactive3d = use3dEngine && !stream && !clipUrl;

  return (
    <div
      data-testid="viewport-frame"
      data-source-mode={interactive3d ? 'engine3d' : stream ? 'capture' : clipUrl ? 'clip' : 'sample'}
      className="relative w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950"
    >
      {/* Hidden media sources feeding WebGL2 */}
      <video ref={videoRef} muted playsInline className="hidden" />
      <img
        ref={sampleImgRef}
        src={sampleFrameUrl}
        alt="Gameplay sample source"
        referrerPolicy="no-referrer"
        className="hidden"
        onLoad={() => setSampleReady(true)}
        onError={() => setSampleFailed(true)}
      />
      {clipUrl && !clipIsVideo && (
        <img
          ref={clipImgRef}
          src={clipUrl}
          alt="Uploaded gameplay frame"
          referrerPolicy="no-referrer"
          className="hidden"
        />
      )}

      {glError ? (
        <div className="flex aspect-video w-full flex-col items-center justify-center gap-2 p-8 text-center">
          <p className="text-sm font-medium text-rose-400">WebGL2 pipeline unavailable</p>
          <p className="max-w-md text-xs text-slate-400">{glError}</p>
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          data-testid="viewport-canvas"
          width={1280}
          height={720}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          className={`block aspect-video w-full bg-slate-950 object-contain ${
            interactive3d ? 'touch-none cursor-grab active:cursor-grabbing' : ''
          }`}
        />
      )}

      {!interactive3d && !sampleReady && sampleFailed && !stream && !clipUrl && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 p-6 text-center">
          <p className="text-sm font-medium text-slate-200">No active video source</p>
          <p className="mt-1 text-xs text-slate-400">
            Click Start to share a game window, or load a gameplay video/image clip.
          </p>
        </div>
      )}
    </div>
  );
}
