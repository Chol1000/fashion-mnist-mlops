import { createContext, useContext, useState, useEffect, type ReactNode } from "react";

type Mode = "light" | "dark";

const ThemeContext = createContext<{ mode: Mode; toggle: () => void }>({
  mode: "light",
  toggle: () => {},
});

const STORAGE_KEY = "fmnist-theme";

function initialMode(): Mode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* private mode / blocked storage — fall through to the OS preference */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(initialMode);

  // AntD's own components pick up dark mode from the ConfigProvider algorithm
  // in App.tsx, but plain CSS (card borders, chart axis colours, the
  // confidence palette) has no way to know the mode unless something exposes
  // it on the DOM. This is that hook — the same attribute index.html's
  // no-flash script sets before React mounts.
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
  }, [mode]);

  const toggle = () => {
    const next: Mode = mode === "light" ? "dark" : "light";
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* choice just won't persist across reloads */
    }
    setMode(next);
  };

  return <ThemeContext.Provider value={{ mode, toggle }}>{children}</ThemeContext.Provider>;
}

export function useThemeMode() {
  return useContext(ThemeContext);
}
