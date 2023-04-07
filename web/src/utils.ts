/**
 * convert file object to byte array
 * @param {File} file target file
 * @returns {Promise<number[]>} a promise object that resolves array contains byte as number
 */
export const convertFileToByteArray = (file: File) => {
  return new Promise<number[]>((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = (e) => {
      if (e.target?.readyState !== FileReader.DONE) {
        return;
      }
      const arrayBuffer = <ArrayBuffer> e.target.result;
      // buffer to byte array
      const typedArray = new Uint8Array(arrayBuffer);
      const byteArray = <number[]> Array.prototype.slice.call(typedArray);
      resolve(byteArray);
    };
    reader.onerror = (e) => {
      reject(e);
    };
  });
};
