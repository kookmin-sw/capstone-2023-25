/**
 * External modules
 */
import styled from "styled-components";
import { space } from "styled-system";

/**
 * Type modules
 */
import type { SpaceProps } from "styled-system";

export const Button  = styled.button<SpaceProps>`
  ${space}
  padding: 8px 12px;
  color: #fff;
  background-color: #4f46e5;
  border: none;
  cursor: pointer;
  outline: none;
  transition: background-color 0.15s ease-out;
  border-radius: 6px;

  :hover {
    background-color: #818cf8;
  }
  :disabled {
    background-color: #a5b4fc;
    cursor: not-allowed;
  }
`;
