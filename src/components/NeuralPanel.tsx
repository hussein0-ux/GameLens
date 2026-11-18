import React, { useRef, useState } from 'react';
import { Cpu, Upload, Trash2, Copy, Check } from 'lucide-react';
import type { PipelineDiagnostics } from '../engine/glPipeline';
import { extractTfliteFromBuffer, readNeuralModelInfo, type NeuralModelInfo } from '../engine/tfliteParser';

interface NeuralPanelProps {
  neuralModel: NeuralModelInfo | null;
  onModelLoaded: (info: NeuralModelInfo | null) => void;
  diagnostics: PipelineDiagnostics | null;
  crashLog: string;
  onCopyAndClearCrash: () => void;
}

export const NeuralPanel: React.FC<NeuralPanelProps> = ({
  neuralModel,
  onModelLoaded,
  diagnostics,
  crashLog,
  onCopyAndClearCrash,
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoadError(null);
    try {
      const buf = await file.arrayBuffer();
      const extracted = await extractTfliteFromBuffer(buf, file.name);
      const info = readNeuralModelInfo(extracted.bytes, extracted.name);
      onModelLoaded(info);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err));
    } finally {
      e.target.value = '';
    }
  };

  const handleCopyCrash = () => {
    onCopyAndClearCrash();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Neural Model Inspector */}
      <section className="rounded-lg border border-[#232C3D] bg-[#121721] p-6 lg:col-span-7">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-100">
              Neural Model (Snapdragon NPU / LiteRT)
            </h2>
            <p className="mt-1.5 text-sm text-slate-400">
              Pick the int8 planar 2x <code className="font-mono text-slate-300">.tflite</code> from Qualcomm AI Hub
              (<code className="font-mono text-slate-300">.tflite</code> or the{' '}
              <code className="font-mono text-slate-300">.zip</code>). Its input size must match the game&apos;s
              render size, for example <span className="font-mono text-slate-300">1168×540</span> for a game
              rendering at <span className="font-mono text-slate-300">1170×540</span>.
            </p>
          </div>
          <Cpu className="h-5 w-5 shrink-0 text-blue-400" />
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".tflite,.zip"
          onChange={handleFile}
          className="hidden"
        />

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-xs font-medium text-white hover:bg-blue-500 transition-colors"
          >
            <Upload className="h-3.5 w-3.5" />
            Pick neural model (.tflite / .zip)
          </button>
          {neuralModel && (
            <button
              type="button"
              onClick={() => onModelLoaded(null)}
              className="inline-flex items-center gap-1.5 rounded-md border border-[#2A354A] bg-[#181F2C] px-3 py-2 text-xs font-medium text-slate-300 hover:text-white"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear model
            </button>
          )}
        </div>

        {loadError && (
          <div className="mt-4 rounded-md border border-red-500/40 bg-red-950/30 px-4 py-3 text-xs text-red-300">
            Could not load model: {loadError}
          </div>
        )}

        <div className="mt-5 border-t border-[#232C3D] pt-5">
          {neuralModel ? (
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <div>
                <dt className="text-xs text-slate-400">Model File</dt>
                <dd className="mt-1 truncate font-mono text-xs font-medium text-slate-100">
                  {neuralModel.fileName}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Input Tensor (NCHW)</dt>
                <dd className="mt-1 font-mono text-xs font-medium text-slate-100">
                  1 × 3 × {neuralModel.h} × {neuralModel.w}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Output Tensor (INT8)</dt>
                <dd className="mt-1 font-mono text-xs font-medium text-slate-100">
                  1 × 3 × {neuralModel.h * neuralModel.scale} × {neuralModel.w * neuralModel.scale} ({neuralModel.scale}x)
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Input Quantization</dt>
                <dd className="mt-1 font-mono text-xs text-slate-200">
                  scale {neuralModel.inScale.toFixed(6)} · zp {neuralModel.inZp}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Output Quantization</dt>
                <dd className="mt-1 font-mono text-xs text-slate-200">
                  scale {neuralModel.outScale.toFixed(6)} · zp {neuralModel.outZp}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-slate-400">Payload Size</dt>
                <dd className="mt-1 font-mono text-xs text-slate-200">
                  {(neuralModel.fileBytes / (1024 * 1024)).toFixed(2)} MB
                </dd>
              </div>
            </dl>
          ) : (
            <p className="text-xs text-slate-400">
              Model: (none) — Upload a quantized planar NCHW <code className="font-mono">.tflite</code> or{' '}
              <code className="font-mono">.zip</code> package to inspect tensor shapes and quantization parameters.
            </p>
          )}
        </div>
      </section>

      {/* Live Diagnostics & Crash Log */}
      <section className="flex flex-col gap-5 rounded-lg border border-[#232C3D] bg-[#121721] p-6 lg:col-span-5">
        <div>
          <h2 className="font-display text-base font-semibold text-slate-100">
            Pipeline Diagnostics &amp; Auto Governor
          </h2>
          <p className="mt-1.5 text-xs text-slate-400">
            Auto mode stays on neural while finishing ≥ 110 results/s, drops to GPU after 2 slow seconds in a row,
            and returns after 5 fast seconds (≥ 118 results/s).
          </p>
        </div>

        <div className="rounded-md border border-[#232C3D] bg-[#0B0E14] p-4 font-mono text-xs leading-relaxed text-slate-200">
          <div>{diagnostics?.aiStatus || 'Neural idle'}</div>
          {diagnostics?.aiStats && <div className="mt-1 text-slate-400">{diagnostics.aiStats}</div>}
          {diagnostics?.diagText && <div className="mt-1 text-blue-300">{diagnostics.diagText}</div>}
        </div>

        <div className="border-t border-[#232C3D] pt-4">
          <div className="flex items-center justify-between">
            <h3 className="font-display text-sm font-medium text-slate-200">Recorded Errors</h3>
            {crashLog && (
              <button
                type="button"
                onClick={handleCopyCrash}
                className="inline-flex items-center gap-1.5 rounded border border-[#2A354A] bg-[#181F2C] px-2.5 py-1 text-xs font-medium text-slate-200 hover:bg-[#222B3D]"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                Copy crash log, then clear it
              </button>
            )}
          </div>
          {crashLog ? (
            <pre className="mt-3 max-h-48 overflow-auto rounded-md border border-red-500/30 bg-[#0B0E14] p-3 font-mono text-[11px] text-red-300">
              {crashLog.slice(-1800)}
            </pre>
          ) : (
            <p className="mt-2 text-xs text-slate-500">No runtime or shader errors recorded.</p>
          )}
        </div>
      </section>
    </div>
  );
};
