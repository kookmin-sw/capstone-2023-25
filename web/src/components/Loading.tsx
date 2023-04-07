/**
 * External modules
 */
import React from "react";
import styled, { keyframes } from "styled-components";
import { ImSpinner3 } from "react-icons/im";

const spin = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export const Wrapper = styled.div<{ show: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex: 1;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease-out;
  visibility: ${({ show }) => show ? "visible" : "hidden"};
  opacity: ${({ show }) => show ? 1 : 0};
  background: rgba(255, 255, 255, 0.6);
  z-index: 10;

  > span {
    margin-left: 8px;
    font-size: 20px;
    color: #3e3e3e;
  }

  > svg {
    animation: ${spin} 0.5s infinite linear;
  }
`;

interface Props {
  show?: boolean;
  target?: string;
}

export const Loading = (props: Props) => {
  const { show = false, target = "" } = props;
  return (
    <Wrapper show={show}>
      <ImSpinner3 size={24} />
      <span>Loading...{target ? ` (${target})` : ""}</span>
    </Wrapper>
  )
};
