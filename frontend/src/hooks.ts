import { useCallback, useEffect, useRef, useState } from "react";

export type Async<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

/**
 * Fetch-on-mount with a manual reload, used by every read-only page.
 *
 * Deliberately thin rather than a query library: there are seven pages and one
 * backend, and the only shared concern is not writing into an unmounted
 * component when a Space cold-starts slowly enough that the user has navigated
 * away before the response lands.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fn()
      .then((d) => {
        if (alive.current) setData(d);
      })
      .catch((e: unknown) => {
        if (alive.current) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive.current) setLoading(false);
      });
    // `fn` is recreated every render by design — callers pass an inline
    // closure — so the dependency list the caller supplies is the real one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}

/** "1h 04m 12s" from a second count — uptime reads better than raw seconds
 *  once a Space has been up for a while. */
export function formatUptime(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
}

export const pct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;
