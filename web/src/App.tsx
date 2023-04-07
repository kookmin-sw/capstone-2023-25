/**
 * External modules
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRecoilState } from "recoil";
import styled from "styled-components";
import {
  alignItems, AlignItemsProps,
  flexDirection, FlexDirectionProps,
  space, SpaceProps
} from "styled-system";

/**
 * Internal modules
 */
import * as PDFiumLoader from "./libs/pdfium-loader";
import { Button, Loading, PDFLayoutSwitch } from "./components";
import { PDF } from "./modules/pdf";
import { loadingState } from "./states/loading";
import { convertFileToByteArray } from "./utils";

/**
 * Type modules
 */
import type { PDFLayout, pDFLayoutChangeHandler } from "./components/PDFLayout";

const Wrapper = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;

  main {
    min-width: 720px;
    flex: 1;
    display: flex;
    flex-direction: column;
    align-tems: center;
  }
`;

const Section = styled.section<AlignItemsProps & FlexDirectionProps & SpaceProps>`
  display: flex;
  ${alignItems}
  ${flexDirection}
  ${space}
`;

const SectionText = styled.h3`
  font-size: 24px;
  line-height: 36px;
  font-weight: 500;
  margin-top: 22px;
  margin-bottom: 8px;
`;

const PDFInputWrapper = styled.div`
  display: flex;
  align-items: baseline;

  > input[type="file"] {
    display: none;
  }

  > span {
    margin-left: 12px;
    font-size: 14px;
  }
`;

const BorderCanvas = styled.canvas`
  border: 2px solid #000000;
`;

function App() {
  const pageCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const pageContainerRef = useRef<HTMLElement>(null);
  const pdfInstance = useRef<PDF>();
  const [loading, setLoading] = useRecoilState(loadingState);
  const [libLoaded, setLibLoaded] = useState(false);
  const [filename, setFilename] = useState("");
  const [PDFSelected, setPDFSelected] = useState(false);
  const [pageCount, setPageCount] = useState(0);
  const [PDFLayout, setPDFLayout] = useState<PDFLayout>("linear");
  // create page canvas along the pageCount
  const pageCanvases = useMemo(() => {
    pageCanvasRefs.current = Array.from({ length: pageCount });
    const pages = Array.from({ length: pageCount }).map((_, i) => {
      return <BorderCanvas ref={(el) => pageCanvasRefs.current[i] = el} key={`page-${i}`} />
    });
    return pages;
  }, [pageCount]);

  /**
   * Library load handler
   */
  const handleLoadLibClick = useCallback(() => {
    if (window.FPDF._PDFium_Init) {
      setLoading({
        status: true,
        target: "Init PDFium",
      });
      window.FPDF._PDFium_Init();
      setLoading({
        status: false,
        target: "Init PDFium",
      });
      setLibLoaded(true);
    }
  }, []);

    /**
   * Library unload handler
   */
  const handleUnloadLibClick = useCallback(() => {
    if (window.FPDF._FPDF_DestroyLibrary) {
      setLoading({
        status: true,
        target: "Destroy PDFium",
      });
      window.FPDF._FPDF_DestroyLibrary();
      setLoading({
        status: false,
        target: "Destroy PDFium",
      });
      setLibLoaded(false);
    }
  }, []);

  /**
   * File select handler
   */
  const handleFileChange = useCallback<React.ChangeEventHandler<HTMLInputElement>>(async (e) => {
    const pdfFile = e.target.files?.item(0);
    if (!pdfFile) return;
    setFilename(pdfFile.name ?? "");

    setLoading({
      status: true,
      target: "File Binary Data"
    });
    try {
      const byteArray = await convertFileToByteArray(pdfFile);
      console.log("Creating PDF instance");
      pdfInstance.current = new PDF(filename, byteArray);
      setPDFSelected(true);
    } catch (err) {
      console.log(err);
    } finally {
      setLoading({
        status: false,
        target: "File Binary Data"
      });
    }
  }, []);

  /**
   * Open PDF button handler
   */
  const handleOpenPDF = useCallback(() => {
    if (!pdfInstance.current) {
      console.warn("No PDF instance to open document.");
      return;
    }
    setLoading({
      status: true,
      target: "Open document",
    });
    try {
      console.log("Opening PDF document...");
      pdfInstance.current.open();
      console.log("PDF successfully opened!");
      const pageCount = pdfInstance.current.getPageCount();
      setPageCount(pageCount);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading({
        status: false,
        target: "Open document",
      });
    }
  }, []);

  /**
   * Close PDF button handler
   */
  const handleClosePDF = useCallback(() => {
    if (!pdfInstance.current) {
      console.warn("No PDF instance to close document.");
      return;
    }
    setLoading({
      status: true,
      target: "Close document",
    });
    try {
      console.log("Closing PDF...");
      pdfInstance.current?.close();
      console.log("PDF successfully closed!");
    } catch (err) {
      console.error(err);
    } finally {
      setLoading({
        status: false,
        target: "Close document",
      });
    }
  }, []);

    /**
   * Load pages button handler
   */
  const handleLoadPages = useCallback(() => {
    pdfInstance.current?.loadPage();
  }, []);

    /**
   * Close pages button handler
   */
  const handleClosePages = useCallback(() => {
    pdfInstance.current?.closePage();
  }, []);
  const handlePDFLayoutChange = useCallback<pDFLayoutChangeHandler>((layout) => {
    setPDFLayout(layout);
  }, []);
  const handlePDFiumError = () => {
    console.error("PDFium error occurred");
    // something to do
  };

  useEffect(() => {
    if (!libLoaded) {
      setLoading({
        status: true,
        target: "PDFium module",
      });
      PDFiumLoader.init({
        onError: handlePDFiumError,
      }).then((result) => {
        if (result) {
          setLoading({
            status: false,
            target: "PDFium module",
          });
        }
      });
    }
  }, [libLoaded]);

  return (
    <Wrapper>
      <header>
        <h1>Beauty of 린냥</h1>
      </header>
      <main>
        <SectionText>Library Load / Unload</SectionText>
        <Section>
          <Button disabled={libLoaded} onClick={handleLoadLibClick}>Load Library</Button>
          <Button ml="6px" disabled={!libLoaded} onClick={handleUnloadLibClick}>Unload Library</Button>
        </Section>

        <SectionText>Choose PDF file to load</SectionText>
        <PDFInputWrapper>
          <Button>
            <label htmlFor="pdf_input">{PDFSelected ? "Re-select PDF" : "Select PDF"}</label>
          </Button>
          <input type="file" name="pdf_input" id="pdf_input" accept=".pdf" onChange={handleFileChange} />
          <span>{filename}</span>
        </PDFInputWrapper>
        <Section mt="24px">
          <Button disabled={!libLoaded || !PDFSelected} onClick={handleOpenPDF}>Open PDF</Button>
          <Button ml="6px" disabled={!libLoaded || !PDFSelected} onClick={handleClosePDF}>Close PDF</Button>
        </Section>
        <Section mt="24px">
          <Button disabled={!libLoaded || !PDFSelected} onClick={handleLoadPages}>Load Pages</Button>
          <Button ml="6px" disabled={!libLoaded || !PDFSelected} onClick={handleClosePages}>Close Pages</Button>
        </Section>
        <SectionText>Output</SectionText>
        <Section mb="8px" alignItems="center">
          <PDFLayoutSwitch mr="16px" layout={PDFLayout} onChange={handlePDFLayoutChange} />
          <span>Pages: {pageCount}</span>
        </Section>
        <Section flexDirection="column" ref={pageContainerRef}>
          {pageCanvases}
        </Section>
      </main>
      <Loading show={loading.status} target={loading.target} />
    </Wrapper>
  );
}

export default App;
