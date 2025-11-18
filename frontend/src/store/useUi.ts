import { create } from "zustand";

interface UiState {
  helpOpen: boolean;
  setHelpOpen: (value: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  helpOpen: false,
  setHelpOpen: (value) => set({ helpOpen: value }),
}));
