interface InitOptions {
  onError: () => void;
  resolve: (result: boolean) => void;
}

const generateModuleWithOptions = (options: Partial<InitOptions>) => {
  const Module: ModuleOptions = {
    print: (message: string) => {
      console.log("Module print:", message);
    },
    printErr: (err: any) => {
      console.error("[PDFium Error]", err);
    },
    onAbort: () => {
      console.error("Failed to load PDFium module.");
      if (options.onError) {
        options.onError();
      }
    },
    onRuntimeInitialized: () => {
      if (options.resolve) {
        options.resolve(true);
      }
    },
    // logReadFiles: true,
    totalDependencies: 0,
    setStatus: (text) => {
      if (!Module.setStatus.last) Module.setStatus.last = { time: Date.now(), text: "" };
      if (text === Module.setStatus.last.text) return;
      var m = text.match(/([^(]+)\((\d+(\.\d+)?)\/(\d+)\)/);
      var now = Date.now();
      if (m && now - Module.setStatus.last.time < 30) return; // if this is a progress update, skip it if too soon
      Module.setStatus.last.time = now;
      Module.setStatus.last.text = text;
      if (m) {
        text = m[1];
        // progressElement.value = parseInt(m[2]) * 100;
        // progressElement.max = parseInt(m[4]) * 100;
        // progressElement.hidden = false;
        // spinnerElement.hidden = false;
      } else {
        // progressElement.value = null;
        // progressElement.max = null;
        // progressElement.hidden = true;
        // if (!text) spinnerElement.style.display = "none";
      }
      // statusElement.innerHTML = text;
    },
    monitorRunDependencies: (left: number) => {
      Module.totalDependencies = Math.max(Module.totalDependencies, left);
      Module.setStatus(left ? 'Preparing... (' + (Module.totalDependencies - left) + '/' + Module.totalDependencies + ')' : 'All downloads complete.');
    },
  };

  return Module;
};

const PDFIUM_SRC = process.env.PUBLIC_URL + process.env.REACT_APP_LIB_PDFIUM_DIR;

const init = (options: Partial<InitOptions> = {}) => {
  return new Promise<boolean>((resolve, reject) => {
    const loaded = document.body.querySelectorAll<HTMLScriptElement>(`script[src="${PDFIUM_SRC}"]`)
    if (loaded.length > 0) {
      console.warn("PDFium script already mounted on document");
      resolve(false);
      return;
    }

    const Module = generateModuleWithOptions({ ...options, resolve });

    // Load PDFium Library from static file
    const pdfiumScript = document.createElement("script");
    document.body.appendChild(pdfiumScript);
    pdfiumScript.onload = () => {
      window.PDFiumModule(Module).then((FPDFModule) => {
        // register PDFium module to window
        window.FPDF = FPDFModule;
        window.Module = FPDFModule;
      }).catch((err) => reject(err));
    };
    pdfiumScript.src = PDFIUM_SRC;
    pdfiumScript.type = "text/javascript";
  });
};

export { init };
