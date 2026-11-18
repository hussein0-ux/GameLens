/**
 * Decides, once per second, whether Auto mode shows the neural picture or the GPU picture.
 * Leaves neural after downSeconds slow seconds in a row; returns only after upSeconds fast seconds in a row,
 * so it does not flicker between the two looks.
 */
export class NeuralGovernor {
  readonly minFps: number;
  readonly downSeconds: number;
  readonly upFps: number;
  readonly upSeconds: number;

  private _useNeural = true;
  private slow = 0;
  private fast = 0;

  constructor(
    minFps = 110,
    downSeconds = 2,
    upFps = 118,
    upSeconds = 5,
  ) {
    this.minFps = minFps;
    this.downSeconds = downSeconds;
    this.upFps = upFps;
    this.upSeconds = upSeconds;
  }

  get useNeural(): boolean {
    return this._useNeural;
  }

  /** neuralFps is how many neural frames finished during the last second. */
  onSecond(neuralFps: number): void {
    if (this._useNeural) {
      this.slow = neuralFps < this.minFps ? this.slow + 1 : 0;
      if (this.slow >= this.downSeconds) {
        this._useNeural = false;
        this.fast = 0;
      }
    } else {
      this.fast = neuralFps >= this.upFps ? this.fast + 1 : 0;
      if (this.fast >= this.upSeconds) {
        this._useNeural = true;
        this.slow = 0;
      }
    }
  }

  reset(): void {
    this._useNeural = true;
    this.slow = 0;
    this.fast = 0;
  }
}
