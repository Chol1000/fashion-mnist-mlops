import {
  createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode,
} from "react";
import { api } from "../api";
import type { Health } from "../types";

export type BackendState = {
  /** null until the very first probe resolves — "unknown", not "down". */
  online: boolean | null;
  health: Health | null;
  /** Probes since the API was last reachable. Drives the wake-up screen copy. */
  attempts: number;
  /** Seconds since the first failed probe in the current outage, or 0. */
  downForSec: number;
  /** True once the API has answered at least once this session. After that an
   *  outage is treated as a blip rather than a cold start, so the app keeps
   *  rendering instead of throwing the user back to a splash screen. */
  everOnline: boolean;
  refresh: () => void;
};

const BackendContext = createContext<BackendState>({
  online: null,
  health: null,
  attempts: 0,
  downForSec: 0,
  everOnline: false,
  refresh: () => {},
});

/** Poll fast while we're waiting for a sleeping Space to come up, slowly once
 *  it's serving — a 20s heartbeat is plenty to notice it going away again. */
const POLL_DOWN_MS = 4_000;
const POLL_UP_MS = 20_000;

export function BackendProvider({ children }: { children: ReactNode }) {
  const [online, setOnline] = useState<boolean | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [attempts, setAttempts] = useState(0);
  const [everOnline, setEverOnline] = useState(false);
  const [downForSec, setDownForSec] = useState(0);
  const downSince = useRef<number | null>(null);

  const probe = useCallback(async () => {
    try {
      const h = await api.health();
      downSince.current = null;
      setHealth(h);
      setOnline(true);
      setEverOnline(true);
      setAttempts(0);
      setDownForSec(0);
    } catch {
      if (downSince.current === null) downSince.current = Date.now();
      setOnline(false);
      setAttempts((a) => a + 1);
    }
  }, []);

  // One interval that re-arms itself at the rate matching the current state,
  // rather than two effects racing each other across a status change.
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const tick = async () => {
      await probe();
      if (cancelled) return;
      timer = window.setTimeout(tick, online ? POLL_UP_MS : POLL_DOWN_MS);
    };
    tick();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // `online` is intentionally a dependency: flipping state restarts the loop
    // at the other cadence.
  }, [probe, online]);

  // Separate 1s ticker so the wake-up screen's elapsed counter moves smoothly
  // instead of jumping by the poll interval.
  useEffect(() => {
    if (online !== false) return;
    const t = setInterval(() => {
      if (downSince.current !== null) {
        setDownForSec(Math.round((Date.now() - downSince.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(t);
  }, [online]);

  return (
    <BackendContext.Provider
      value={{ online, health, attempts, downForSec, everOnline, refresh: probe }}
    >
      {children}
    </BackendContext.Provider>
  );
}

export function useBackend() {
  return useContext(BackendContext);
}
