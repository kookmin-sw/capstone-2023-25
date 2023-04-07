/**
 * External modules
 */
import { atom } from "recoil";

export const loadingState = atom({
  key: "loading",
  default: {
    status: false,
    target: "",
  },
});
