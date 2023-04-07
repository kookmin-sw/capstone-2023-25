export const fillValue = (memory: Uint8Array, size: number, offset: number, value: number) => {
  for (let i = 0; i < size; i++) {
    memory[offset + i] = value;
  }
};
