import { describe, expect, it } from 'vitest';
import { NeuralGovernor } from '../neuralGovernor';

function feed(g: NeuralGovernor, ...fps: number[]) {
  for (const f of fps) g.onSecond(f);
}

describe('NeuralGovernor', () => {
  it('starts on neural and stays while fast', () => {
    const g = new NeuralGovernor();
    feed(g, 119, 120, 118, 119);
    expect(g.useNeural).toBe(true);
  });

  it('one slow second does not switch', () => {
    const g = new NeuralGovernor();
    feed(g, 100, 119, 100, 119);
    expect(g.useNeural).toBe(true);
  });

  it('two slow seconds in a row switch to GPU', () => {
    const g = new NeuralGovernor();
    feed(g, 100, 105);
    expect(g.useNeural).toBe(false);
  });

  it('returns to neural only after five fast seconds', () => {
    const g = new NeuralGovernor();
    feed(g, 90, 90);
    feed(g, 120, 120, 120, 120);
    expect(g.useNeural).toBe(false);
    feed(g, 120);
    expect(g.useNeural).toBe(true);
  });

  it('a slow second resets recovery', () => {
    const g = new NeuralGovernor();
    feed(g, 90, 90);
    feed(g, 120, 120, 120, 120, 110, 120, 120, 120, 120);
    expect(g.useNeural).toBe(false);
    feed(g, 120);
    expect(g.useNeural).toBe(true);
  });

  it('after returning it needs two new slow seconds to leave again', () => {
    const g = new NeuralGovernor();
    feed(g, 90, 90, 120, 120, 120, 120, 120);
    expect(g.useNeural).toBe(true);
    feed(g, 100);
    expect(g.useNeural).toBe(true);
    feed(g, 100);
    expect(g.useNeural).toBe(false);
  });
});
