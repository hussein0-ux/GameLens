import { unzipSync } from 'fflate';

export interface NeuralModelInfo {
  name: string;
  sizeBytes: number;
  w: number;
  h: number;
  scale: number;
  inScale: number;
  inZp: number;
  outScale: number;
  outZp: number;
}

const TENSOR_TYPE_INT8 = 9;
const TENSOR_TYPE_NAMES: Record<number, string> = {
  0: 'FLOAT32',
  1: 'FLOAT16',
  2: 'INT32',
  3: 'UINT8',
  4: 'INT64',
  5: 'STRING',
  6: 'BOOL',
  7: 'INT16',
  8: 'COMPLEX64',
  9: 'INT8',
};

function readFieldOffset(dv: DataView, tablePos: number, fieldIndex: number): number {
  const vtableRel = dv.getInt32(tablePos, true);
  const vtablePos = tablePos - vtableRel;
  const vtableSize = dv.getUint16(vtablePos, true);
  const entryOffset = 4 + fieldIndex * 2;
  if (entryOffset >= vtableSize) return 0;
  const fieldRel = dv.getUint16(vtablePos + entryOffset, true);
  return fieldRel === 0 ? 0 : tablePos + fieldRel;
}

function readIndirectTable(dv: DataView, fieldPos: number): number {
  return fieldPos + dv.getUint32(fieldPos, true);
}

function readVectorStart(dv: DataView, fieldPos: number): { len: number; dataStart: number } {
  const vecPos = fieldPos + dv.getUint32(fieldPos, true);
  const len = dv.getUint32(vecPos, true);
  return { len, dataStart: vecPos + 4 };
}

function readInt32Vector(dv: DataView, tablePos: number, fieldIndex: number): number[] {
  const f = readFieldOffset(dv, tablePos, fieldIndex);
  if (!f) return [];
  const { len, dataStart } = readVectorStart(dv, f);
  const out: number[] = [];
  for (let i = 0; i < len; i++) {
    out.push(dv.getInt32(dataStart + i * 4, true));
  }
  return out;
}

function readFloat32Vector(dv: DataView, tablePos: number, fieldIndex: number): number[] {
  const f = readFieldOffset(dv, tablePos, fieldIndex);
  if (!f) return [];
  const { len, dataStart } = readVectorStart(dv, f);
  const out: number[] = [];
  for (let i = 0; i < len; i++) {
    out.push(dv.getFloat32(dataStart + i * 4, true));
  }
  return out;
}

function readInt64VectorAsNumbers(dv: DataView, tablePos: number, fieldIndex: number): number[] {
  const f = readFieldOffset(dv, tablePos, fieldIndex);
  if (!f) return [];
  const { len, dataStart } = readVectorStart(dv, f);
  const out: number[] = [];
  for (let i = 0; i < len; i++) {
    const low = dv.getUint32(dataStart + i * 8, true);
    const high = dv.getInt32(dataStart + i * 8 + 4, true);
    out.push(high * 4294967296 + low);
  }
  return out;
}

function readTableVectorItem(dv: DataView, tablePos: number, fieldIndex: number, itemIndex: number): number {
  const f = readFieldOffset(dv, tablePos, fieldIndex);
  if (!f) throw new Error(`Missing table vector at field ${fieldIndex}`);
  const { len, dataStart } = readVectorStart(dv, f);
  if (itemIndex < 0 || itemIndex >= len) {
    throw new Error(`Index ${itemIndex} out of bounds (len=${len})`);
  }
  const elemPos = dataStart + itemIndex * 4;
  return elemPos + dv.getUint32(elemPos, true);
}

/**
 * Extracts a .tflite from a .zip if needed, validates the TFL3 FlatBuffer header,
 * and reads NCHW int8 tensor shape & quantisation parameters (matching NeuralStage.kt).
 */
export function parseNeuralModelFile(fileName: string, rawBytes: Uint8Array): NeuralModelInfo {
  let tfliteBytes: Uint8Array | null = null;
  let resolvedName = fileName;

  if (rawBytes.length > 4 && rawBytes[0] === 0x50 && rawBytes[1] === 0x4b) {
    const entries = unzipSync(rawBytes);
    for (const [entryName, data] of Object.entries(entries)) {
      if (!entryName.endsWith('/') && entryName.endsWith('.tflite')) {
        tfliteBytes = data;
        resolvedName = entryName.split('/').pop() || 'model.tflite';
        break;
      }
    }
    if (!tfliteBytes) {
      throw new Error('No .tflite file found in the zip');
    }
  } else {
    tfliteBytes = rawBytes;
  }

  if (
    tfliteBytes.length <= 8 ||
    tfliteBytes[4] !== 0x54 || // 'T'
    tfliteBytes[5] !== 0x46 || // 'F'
    tfliteBytes[6] !== 0x4c || // 'L'
    tfliteBytes[7] !== 0x33    // '3'
  ) {
    throw new Error(`${resolvedName} is not a .tflite model`);
  }

  const dv = new DataView(tfliteBytes.buffer, tfliteBytes.byteOffset, tfliteBytes.byteLength);
  const rootPos = dv.getUint32(0, true);
  const subgraph0 = readTableVectorItem(dv, rootPos, 2, 0);
  const inputs = readInt32Vector(dv, subgraph0, 1);
  const outputs = readInt32Vector(dv, subgraph0, 2);
  if (inputs.length === 0 || outputs.length === 0) {
    throw new Error('Model has no input or output tensors');
  }

  const inTensor = readTableVectorItem(dv, subgraph0, 0, inputs[0]);
  const outTensor = readTableVectorItem(dv, subgraph0, 0, outputs[0]);

  const inShape = readInt32Vector(dv, inTensor, 0);
  const outShape = readInt32Vector(dv, outTensor, 0);

  if (inShape.length !== 4 || outShape.length !== 4 || inShape[1] !== 3 || outShape[1] !== 3) {
    throw new Error(`Neural mode needs a planar NCHW model, got [${inShape.join(', ')}] -> [${outShape.join(', ')}]`);
  }

  const inTypeField = readFieldOffset(dv, inTensor, 1);
  const outTypeField = readFieldOffset(dv, outTensor, 1);
  // In schema.fbs, TensorType FLOAT32 = 0 can be omitted as default value.
  const inType = inTypeField ? dv.getInt8(inTypeField) : 0;
  const outType = outTypeField ? dv.getInt8(outTypeField) : 0;
  if (inType !== TENSOR_TYPE_INT8 || outType !== TENSOR_TYPE_INT8) {
    const inName = TENSOR_TYPE_NAMES[inType] ?? String(inType);
    const outName = TENSOR_TYPE_NAMES[outType] ?? String(outType);
    throw new Error(`Neural mode needs an int8 (w8a8) model, got ${inName} -> ${outName}`);
  }

  const h = inShape[2];
  const w = inShape[3];
  if (w <= 0 || h <= 0 || w % 4 !== 0) {
    throw new Error(`Model width ${w} must be divisible by 4`);
  }
  if (outShape[2] % h !== 0 || outShape[3] % w !== 0 || outShape[2] / h !== outShape[3] / w) {
    throw new Error(`Unsupported output shape [${outShape.join(', ')}]`);
  }

  const readQuant = (tensorPos: number): { scale: number; zp: number } => {
    const qField = readFieldOffset(dv, tensorPos, 4);
    if (!qField) return { scale: 1 / 255, zp: 0 };
    const qTable = readIndirectTable(dv, qField);
    const scales = readFloat32Vector(dv, qTable, 2);
    const zps = readInt64VectorAsNumbers(dv, qTable, 3);
    return {
      scale: scales[0] ?? 1 / 255,
      zp: (zps[0] ?? 0) + 128,
    };
  };

  const inQ = readQuant(inTensor);
  const outQ = readQuant(outTensor);

  return {
    name: resolvedName,
    sizeBytes: tfliteBytes.byteLength,
    w,
    h,
    scale: outShape[2] / h,
    inScale: inQ.scale,
    inZp: inQ.zp,
    outScale: outQ.scale,
    outZp: outQ.zp,
  };
}
