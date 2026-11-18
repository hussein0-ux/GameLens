/**
 * Decides, once per second, whether Auto mode shows the neural picture or the GPU picture.
 * Leaves neural after `downSeconds` slow seconds in a row; returns only after `upSeconds` fast seconds in a row,
 * so it does not flicker between the two looks.
 */
export class NeuralGovernor {
  useNeural = true;
  private slow = 0;
  private fast = 0;

  constructor(
    private readonly minFps = 110,
    private readonly downSeconds = 2,
    private readonly upFps = 118,
    private readonly upSeconds = 5,
  ) {}

  /** `neuralFps` is how many neural frames finished during the last second. */
  onSecond(neuralFps: number): void {
    if (this.useNeural) {
      this.slow = neuralFps < this.minFps ? this.slow + 1 : 0;
      if (this.slow >= this.downSeconds) {
        this.useNeural = false;
        this.fast = 0;
      }
    } else {
      this.fast = neuralFps >= this.upFps ? this.fast + 1 : 0;
      if (this.fast >= this.upSeconds) {
        this.useNeural = true;
        this.slow = 0;
      }
    }
  }
}
