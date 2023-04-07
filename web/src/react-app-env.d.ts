/* eslint-disable @typescript-eslint/no-explicit-any */
/// <reference types="react-scripts" />

interface ModuleOptions {
  [key: string]: any;
  print: (message: string) => void;
  printErr: (message: string) => void;
  onAbort: () => void;
  onRuntimeInitialized: () => void;
  totalDependencies: number;
  setStatus: {
    (text: string): void;
    last?: {
      time: number;
      text: string;
    };
  };
  monitorRunDependencies: (left: number) => void;
}

interface FPDF extends ModuleOptions {
  HEAP8: Int8Array;
  HEAP16: Int16Array;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  HEAPU16: Uint16Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
  asm: {
    [key: string]: any;
    malloc: (size: number) => number;
    free: (address: number) => number;
  };
  ccall: (identifier: string, returnType?: any, argTypes?: any, args?: any, options?: any) => any;
  cwrap: (identifier: string, returnType?: any, argTypes?: any, args?: any, options?: any) => any;
  // Library
  _PDFium_Init: () => void;
  _FPDF_InitLibrary: () => void;
  _FPDF_InitLibraryWithConfig: (...args: any) => void;
  _FPDF_DestroyLibrary: () => void;
  // Document
  _FPDF_LoadDocument: (...args: any) => any;
  _FPDF_LoadMemDocument: (buffer: number, size: number, password: string) => number;
  _FPDF_CloseDocument: (document: number) => void;
  _FPDF_GetPageCount: (document: number) => number;
  // Page
  _FPDF_LoadPage: (document: number, index: number) => number;
  _FPDF_GetPageWidth: (page: number) => number;
  _FPDF_GetPageHeight: (page: number) => number;
  _FPDF_GetPageWidthF: (page: number) => number;
  _FPDF_GetPageHeightF: (page: number) => number;
  _FPDF_GetPageSizeByIndex: (document: number, pageIndex: number, width: number, height: number) => number;
  _FPDF_ClosePage: (page: number) => void;
  // Page Object
  _FPDFPage_CountObjects: (page: number) => number;
  _FPDFPage_GetObject: (page: number, index: number) => number;
  _FPDFPage_GenerateContent: (page: number) => boolean;
  _FPDFPageObj_Destroy: (pageObject: number) => void;
  // TextPage
  _FPDFText_LoadPage: (page: number) => number;
  _FPDFText_CountChars: (textPage: number) => number;
  _FPDFText_GetCharBox: (textIndex: number) => void;
  _FPDFText_ClosePage: (textPage: number) => void;
  // Bitmap
  _FPDFBitmap_Create: (width: number, height: number, alpha: number) => number;
  _FPDFBitmap_CreateEx: (width: number, height: number, format: number, firstScan: number, stride: number) => number;
  _FPDFBitmap_FillRect: (
    bitmap: number,
    left: number,
    top: number,
    width: number,
    height: number,
    color: number
  ) => void;
  _FPDFBitmap_Destroy: (bitmap: number) => void;
}

declare namespace NodeJS {
  interface ProcessEnv {
    readonly NODE_ENV: "development" | "production" | "test";
    readonly PUBLIC_URL: string;
    readonly REACT_APP_LIB_PDFIUM_DIR: string;
  }
}

interface Window {
  PDFiumModule: (Module?: object) => Promise<FPDF>;
  Module: FPDF;
  FPDF: FPDF;
}
