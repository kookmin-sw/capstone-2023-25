/**
 * External modules
 */
import React from "react";
import styled from "styled-components";
import { BsGrid, BsViewStacked } from "react-icons/bs";
import { space } from "styled-system";

/**
 * Type modules
 */
import type { SpaceProps } from "styled-system";

const Wrapper = styled.div<SpaceProps>`
  ${space}
  display: flex;

  > button:nth-child(2) {
    margin-left: 6px;
  }
`;

const ImageButton = styled.div<{ selected: boolean }>`
  padding: 8px;
  background-color: ${({ selected }) => selected ? "#4f46e5" : "#fff"};
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid ${({ selected }) => selected ? "#4f46e5" : "#3f3f3f"};
  transition: all 0.15s ease-out;
  cursor: pointer;
`.withComponent("button");

export type PDFLayout = "grid" | "linear";
export type pDFLayoutChangeHandler = (layout: PDFLayout) => void;

interface PDFLayoutSwitchProps extends SpaceProps {
  layout: PDFLayout;
  onChange: pDFLayoutChangeHandler;
}

export const PDFLayoutSwitch = (props: PDFLayoutSwitchProps) => {
  const { layout, onChange, ...styles } = props;

  return (
    <Wrapper {...styles}>
      <ImageButton selected={layout === "linear"} onClick={() => onChange("linear")}>
        <BsViewStacked color={layout === "linear" ? "#fff" : "#000"} />
      </ImageButton>
      <ImageButton selected={layout === "grid"} onClick={() => onChange("grid")}>
        <BsGrid color={layout === "grid" ? "#fff" : "#000"} />
      </ImageButton>
    </Wrapper>
  );
}
