export interface NeuralModelInfo {
  fileName: string;
  fileBytes: number;
  w: number;
  h: number;
  scale: number;
  inScale: number;
  inZp: number;
  outScale: number;
  outZp: number;
}

/**
 * Extracts the first .tflite entry from a ZIP archive (supports Store method 0 and Deflate method 8).
 */
export async function extractTfliteFromBuffer(
  buffer: ArrayBuffer,
  originalName: string,
): Promise<{ name: string; bytes: Uint8Array }> {
  const bytes = new Uint8Array(buffer);
  // Check PK\x03\x04 ZIP magic
  if (bytes.length > 4 && bytes[0] === 0x50 && bytes[1] === 0x4b) {
    const view = new DataView(buffer);
    let offset = 0;
    while (offset + 30 <= bytes.length) {
      const sig = view.getUint32(offset, true);
      if (sig !== 0x04034b50) break;
      const compression = view.getUint16(offset + 8, true);
      const compSize = view.getUint32(offset + 18, true);
      const nameLen = view.getUint16(offset + 26, true);
      const extraLen = view.getUint16(offset + 28, true);
      const nameStart = offset + 30;
      const nameBytes = bytes.slice(nameStart, nameStart + nameLen);
      const entryName = new TextDecoder().decode(nameBytes);
      const dataStart = nameStart + nameLen + extraLen;

      if (!entryName.endsWith('/') && entryName.endsWith('.tflite')) {
        const compressed = bytes.slice(dataStart, dataStart + compSize);
        const fileName = entryName.split('/').pop() || 'model.tflite';
        if (compression === 0) {
          return { name: fileName, bytes: compressed };
        }
        if (compression === 8) {
          if (typeof DecompressionStream === 'undefined') {
            throw new Error('Browser does not support DecompressionStream for deflated .zip files');
          }
          const ds = new DecompressionStream('deflate-raw');
          const writer = ds.writable.getWriter();
          void writer.write(compressed);
          void writer.close();
          const decompressed = await new Response(ds.readable).arrayBuffer();
          return { name: fileName, bytes: new Uint8Array(decompressed) };
        }
        throw new Error(`Unsupported ZIP compression method ${compression} for ${fileName}`);
      }

      if (compSize === 0 && (view.getUint16(offset + 6, true) & 0x08) !== 0) {
        // Fallback: scan for next local file header if data descriptor is used
        let next = dataStart;
        while (next + 4 <= bytes.length) {
          if (view.getUint32(next, true) === 0x04034b50) break;
          next++;
        }
        offset = next;
      } else {
        offset = dataStart + compSize;
      }
    }
    throw new Error('No .tflite file found in the zip');
  }

  return { name: originalName, bytes };
}

// TFLite TensorType enum values (from schema_generated.h)
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

/**
 * Minimal FlatBuffer reader for TFLite Model schema to extract input/output shapes & quantization
 * without needing a native C++/TFLite runtime.
 */
export function readNeuralModelInfo(bytes: Uint8Array, fileName: string): NeuralModelInfo {
  if (
    bytes.length <= 8 ||
    bytes[4] !== 0x54 || // 'T'
    bytes[5] !== 0x46 || // 'F'
    bytes[6] !== 0x4c || // 'L'
    bytes[7] !== 0x33    // '3'
  ) {
    throw new Error(`${fileName} is not a .tflite model`);
  }

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const readFieldOffset = (tablePos: number, fieldIndex: number): number => {
    const vtableOffset = tablePos - dv.getInt32(tablePos, true);
    const vtableSize = dv.getUint16(vtableOffset, true);
    const slot = 4 + fieldIndex * 2;
    if (slot >= vtableSize) return 0;
    const rel = dv.getUint16(vtableOffset + slot, true);
    return rel === 0 ? 0 : tablePos + rel;
  };

  const readIndirect = (pos: number): number => pos + dv.getUint32(pos, true);

  const readIntVector = (fieldPos: number): number[] => {
    if (fieldPos === 0) return [];
    const vecPos = readIndirect(fieldPos);
    const len = dv.getUint32(vecPos, true);
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
      out.push(dv.getInt32(vecPos + 4 + i * 4, true));
    }
    return out;
  };

  const readFloatVector = (fieldPos: number): number[] => {
    if (fieldPos === 0) return [];
    const vecPos = readIndirect(fieldPos);
    const len = dv.getUint32(vecPos, true);
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
      out.push(dv.getFloat32(vecPos + 4 + i * 4, true));
    }
    return out;
  };

  const readLongVector = (fieldPos: number): number[] => {
    if (fieldPos === 0) return [];
    const vecPos = readIndirect(fieldPos);
    const len = dv.getUint32(vecPos, true);
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
      const lo = dv.getUint32(vecPos + 4 + i * 8, true);
      const hi = dv.getInt32(vecPos + 4 + i * 8 + 4, true);
      out.push(hi * 4294967296 + lo);
    }
    return out;
  };

  const rootPos = dv.getUint32(0, true);
  // Model table: field 2 = subgraphs (vector<SubGraph>)
  const subgraphsField = readFieldOffset(rootPos, 2);
  if (subgraphsField === 0) {
    throw new Error(`${fileName} has no subgraphs`);
  }
  const subgraphsVec = readIndirect(subgraphsField);
  const subgraphsLen = dv.getUint32(subgraphsVec, true);
  if (subgraphsLen < 1) {
    throw new Error(`${fileName} has empty subgraphs`);
  }
  const sg0 = readIndirect(subgraphsVec + 4);

  // SubGraph table: field 0 = tensors, field 1 = inputs, field 2 = outputs
  const tensorsField = readFieldOffset(sg0, 0);
  const inputsField = readFieldOffset(sg0, 1);
  const outputsField = readFieldOffset(sg0, 2);
  if (!tensorsField || !inputsField || !outputsField) {
    throw new Error(`${fileName} is missing tensor metadata`);
  }

  const tensorsVec = readIndirect(tensorsField);
  const inputs = readIntVector(inputsField);
  const outputs = readIntVector(outputsField);
  if (inputs.length === 0 || outputs.length === 0) {
    throw new Error(`${fileName} has no input/output tensors`);
  }

  const readTensorInfo = (tensorIdx: number) => {
    const tPos = readIndirect(tensorsVec + 4 + tensorIdx * 4);
    // Tensor table: field 0 = shape, field 1 = type (byte), field 4 = quantization
    const shape = readIntVector(readFieldOffset(tPos, 0));
    const typeField = readFieldOffset(tPos, 1);
    const typeCode = typeField ? dv.getUint8(typeField) : 0;
    const quantField = readFieldOffset(tPos, 4);
    let scale = 1 / 255;
    let zeroPoint = -128;
    if (quantField) {
      const qPos = readIndirect(quantField);
      // QuantizationParameters: field 2 = scale (vector<float>), field 3 = zero_point (vector<long>)
      const scales = readFloatVector(readFieldOffset(qPos, 2));
      const zps = readLongVector(readFieldOffset(qPos, 3));
      if (scales.length > 0) scale = scales[0];
      if (zps.length > 0) zeroPoint = zps[0];
    }
    return { shape, typeCode, typeName: TENSOR_TYPE_NAMES[typeCode] ?? `TYPE_${typeCode}`, scale, zeroPoint };
  };

  const ti = readTensorInfo(inputs[0]);
  const to = readTensorInfo(outputs[0]);

  if (ti.shape.length !== 4 || to.shape.length !== 4 || ti.shape[1] !== 3 || to.shape[1] !== 3) {
    throw new Error(
      `Neural mode needs a planar NCHW model, got [${ti.shape.join(', ')}] -> [${to.shape.join(', ')}]`,
    );
  }
  if (ti.typeCode !== 9 || to.typeCode !== 9) {
    throw new Error(`Neural mode needs an int8 (w8a8) model, got ${ti.typeName} -> ${to.typeName}`);
  }

  const h = ti.shape[2];
  const w = ti.shape[3];
  if (w % 4 !== 0) {
    throw new Error(`Model width ${w} must be divisible by 4`);
  }
  if (to.shape[2] % h !== 0 || to.shape[3] % w !== 0 || to.shape[2] / h !== to.shape[3] / w) {
    throw new Error(`Unsupported output shape [${to.shape.join(', ')}]`);
  }

  return {
    fileName,
    fileBytes: bytes.length,
    w,
    h,
    scale: to.shape[2] / h,
    inScale: ti.scale,
    inZp: ti.zeroPoint + 128,
    outScale: to.scale,
    outZp: to.zeroPoint + 128,
  };
}
