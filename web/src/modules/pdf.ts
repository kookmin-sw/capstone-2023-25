/**
 * Internal modules
 */
import { FPDFBitmapFormat } from "../libs/pdfium-constants";
import * as memory from "../libs/memory";

export class PDF {
  filename: string;
  fileSize: number;
  byteArray: number[];
  memory: {
    binaryAddress: number;
    documentAddress: number;
  };
  pageCount: number;
  opened: boolean;
  pages: PDFPage[] | null;

  constructor(filename: string, byteArray: number[]) {
    this.filename = filename;
    this.byteArray = byteArray;
    this.fileSize = byteArray.length;
    this.memory = {
      binaryAddress: -1,
      documentAddress: -1,
    };
    this.pageCount = 0;
    this.opened = false;
    this.pages = null;
  }

  isLoaded() {
    if (!this.opened || this.memory.binaryAddress < 0 || this.memory.binaryAddress < 0) {
      return false;
    }
    return true;
  }

  open() {
    if (this.isLoaded()) {
      console.warn("PDF document already loaded!");
      return;
    }

    this.memory.binaryAddress = window.FPDF.asm.malloc(this.fileSize);
    window.FPDF.HEAPU8.set(this.byteArray, this.memory.binaryAddress);
    this.memory.documentAddress = window.FPDF._FPDF_LoadMemDocument(this.memory.binaryAddress, this.fileSize, "");
    this.opened = true;
  }

  close() {
    if (!this.isLoaded()) {
      console.warn("PDF document was not loaded!");
      return;
    }

    window.FPDF._FPDF_CloseDocument(this.memory.documentAddress);
    window.FPDF.asm.free(this.memory.binaryAddress);
    this.pageCount = 0;
    this.opened = false;
  }

  loadPage() {
    if (!this.isLoaded()) {
      console.warn("PDF document was not loaded!");
      return;
    }
    if (this.pageCount === 0) {
      this.getPageCount();
    }

    this.pages = Array(this.pageCount);
    for (let i = 0; i < this.pageCount; i++) {
      this.pages[i] = new PDFPage(this.memory.documentAddress, i);
      this.pages[i].load();
      this.pages[i].render();
    }
  }

  closePage() {
    if (!this.isLoaded()) {
      console.warn("PDF document was not loaded!");
      return;
    }
    if (this.pageCount === 0) {
      console.warn("There are no pages to close.");
      return;
    }
    for (let i = 0; i < this.pageCount; i++) {
      this.pages?.[i].close();
    }
  }

  getPageCount() {
    if (!this.isLoaded()) {
      console.warn("PDF document was not loaded!");
      return -1;
    }

    if (this.pageCount < 1) {
      this.pageCount = window.FPDF._FPDF_GetPageCount(this.memory.documentAddress);
    }

    return this.pageCount;
  }
}

export class PDFPage {
  pageIndex: number;
  memory: {
    documentAddress: number;
    pageAddress: number;
    bitmapAddress: number;
    bitmapBufferAddress: number;
  };
  size: {
    width: number;
    height: number;
  };
  loaded: boolean;

  constructor(documentAddress: number, pageIndex: number) {
    this.pageIndex = pageIndex;
    this.memory = {
      documentAddress,
      pageAddress: -1,
      bitmapAddress: -1,
      bitmapBufferAddress: -1,
    };
    this.size = {
      width: 0,
      height: 0,
    };
    this.loaded = false;
  }

  isLoaded() {
    if (!this.isLoaded || this.memory.documentAddress < 0 || this.memory.pageAddress < 0) {
      return false;
    }
    return true;
  }

  load() {
    if (this.isLoaded()) {
      console.warn(`PDF Page index: ${this.pageIndex} already loaded!`);
      return;
    }

    this.memory.pageAddress = window.FPDF._FPDF_LoadPage(this.memory.documentAddress, this.pageIndex);
    // const result = window.FPDF._FPDF_GetPageSizeByIndex(
    //   this.memory.documentAddress,
    //   this.pageIndex,
    //   this.size.width,
    //   this.size.height
    // );
    // if (result === 0) {
    //   console.warn(`Failed to call GetPageSizeByIndex index: ${this.pageIndex} code: ${result}`);
    // } else {
    //   console.log("page size:", this.size);
    // }
    this.size.width = window.FPDF._FPDF_GetPageWidthF(this.memory.pageAddress);
    this.size.height = window.FPDF._FPDF_GetPageHeightF(this.memory.pageAddress);
    console.log(this.size);
    this.loaded = true;
  }

  close() {
    if (!this.isLoaded()) {
      console.warn(`PDF Page index: ${this.pageIndex} was not loaded!`);
      return;
    }

    window.FPDF._FPDF_ClosePage(this.memory.pageAddress);
  }

  render() {
    const STRIDE = 4;
    const memorySize = this.size.width * this.size.height * STRIDE;
    const address = window.FPDF.asm.malloc(memorySize);
    memory.fillValue(window.FPDF.HEAPU8, memorySize, address, 0);
    this.memory.bitmapAddress = window.FPDF._FPDFBitmap_CreateEx(
      this.size.width,
      this.size.height,
      FPDFBitmapFormat.FPDFBitmap_BGRA,
      this.memory.bitmapBufferAddress,
      this.size.width * STRIDE
    );
    window.FPDF._FPDFBitmap_FillRect(this.memory.bitmapAddress, 0, 0, this.size.width, this.size.height, 0x00000000);
  }
}

// 2482 / 1755
