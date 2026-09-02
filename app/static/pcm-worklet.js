class PCMCollector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    let index = 0;
    while (index < input.length) {
      const available = this.buffer.length - this.offset;
      const count = Math.min(available, input.length - index);
      this.buffer.set(input.subarray(index, index + count), this.offset);
      this.offset += count;
      index += count;
      if (this.offset === this.buffer.length) {
        const output = this.buffer;
        this.port.postMessage(output, [output.buffer]);
        this.buffer = new Float32Array(2048);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-collector', PCMCollector);
