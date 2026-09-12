import { useEffect, useRef, type ReactNode } from "react";
import {
  Card, Col, Row, Statistic, Tag, Typography, Space, Tooltip,
  Spin as AntSpin, Result as AntResult, Button as AntButton, Grid,
} from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useThemeMode } from "./context/ThemeContext";

const { Title, Text } = Typography;

/* ── Confidence palette ──────────────────────────────────────────────────────
   Two palettes rather than one: the light-mode colours (deep green through
   crimson) are chosen for contrast against white and go muddy on AntD's
   #141414 dark surface. The dark set is the same three hues lifted in
   lightness. Charts need literal hex (recharts can't resolve CSS vars), so
   this is the single source both the DOM and the charts read from. */
const CONF_LIGHT = { high: "#15803d", mid: "#b45309", low: "#be123c" } as const;
const CONF_DARK = { high: "#4ade80", mid: "#fbbf24", low: "#f87171" } as const;

export type ConfLevel = "high" | "mid" | "low";

/** The thresholds the original dashboard used, kept so the colour a user sees
 *  for a given prediction doesn't change between the two UIs. */
export function confLevel(confidence: number): ConfLevel {
  if (confidence >= 0.85) return "high";
  if (confidence >= 0.5) return "mid";
  return "low";
}

export function useConfColors() {
  const { mode } = useThemeMode();
  return mode === "dark" ? CONF_DARK : CONF_LIGHT;
}

export function useConfColor(confidence: number) {
  return useConfColors()[confLevel(confidence)];
}

const CONF_TAG_COLOR: Record<ConfLevel, string> = {
  high: "green",
  mid: "gold",
  low: "red",
};

export function ConfidenceTag({ confidence }: { confidence: number }) {
  const level = confLevel(confidence);
  const wording = level === "high" ? "High confidence" : level === "mid" ? "Moderate confidence" : "Low confidence";
  return (
    <Tag color={CONF_TAG_COLOR[level]} style={{ marginInlineEnd: 0, fontWeight: 600 }}>
      {wording}
    </Tag>
  );
}

/** Axis/grid/tooltip colours for recharts, which renders to SVG attributes and
 *  so can't inherit any of AntD's theming. Keeps every chart switching with
 *  the theme instead of keeping light-mode gridlines on a dark card. */
export function useChartTheme() {
  const { mode } = useThemeMode();
  const dark = mode === "dark";
  return {
    dark,
    axis: dark ? "#8c8c8c" : "#8792a2",
    grid: dark ? "#303030" : "#e8ecf2",
    accent: dark ? "#7c84f0" : "#4338ca",
    accent2: dark ? "#4ade80" : "#0d9488",
    muted: dark ? "#3a3a3a" : "#e5e7eb",
    surface: dark ? "#1f1f1f" : "#ffffff",
    text: dark ? "#e6e6e6" : "#1b2030",
    tooltip: {
      fontSize: 12,
      borderRadius: 6,
      background: dark ? "#1f1f1f" : "#ffffff",
      border: `1px solid ${dark ? "#303030" : "#e3e6ef"}`,
      color: dark ? "#e6e6e6" : "#1b2030",
    } as const,
  };
}

/** Page masthead: eyebrow + title + one line of context, with an optional
 *  action slot on the right. Every page opens with this so they read as one
 *  product rather than seven separately-styled screens. */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  extra,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        flexWrap: "wrap",
        gap: 12,
        marginBottom: 20,
      }}
    >
      <div style={{ minWidth: 0 }}>
        {eyebrow && (
          <Text
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--color-primary)",
            }}
          >
            {eyebrow}
          </Text>
        )}
        <Title level={3} style={{ margin: eyebrow ? "6px 0 0" : 0 }}>
          {title}
        </Title>
        {subtitle && (
          <Text type="secondary" style={{ fontSize: 13, display: "block", marginTop: 6, maxWidth: 760 }}>
            {subtitle}
          </Text>
        )}
      </div>
      {/* The action slot must be allowed to shrink and wrap: on a phone a
          fixed-width group here is wider than the viewport, and
          `flex-shrink: 0` would push the whole page sideways rather than
          letting the group wrap onto its own line. */}
      {extra && <div style={{ minWidth: 0, maxWidth: "100%" }}>{extra}</div>}
    </div>
  );
}

/** A single KPI tile. Wraps AntD's Statistic so every stat in the app gets the
 *  same card shell, icon treatment, and optional "what is this" tooltip. */
export function StatCard({
  label,
  value,
  suffix,
  precision,
  icon,
  color,
  hint,
}: {
  label: string;
  value: number | string;
  suffix?: string;
  precision?: number;
  icon?: ReactNode;
  color?: string;
  hint?: string;
}) {
  return (
    <Card size="small" styles={{ body: { padding: "16px 18px" } }}>
      <Statistic
        title={
          <Space size={4}>
            {icon}
            <span style={{ fontSize: 12 }}>{label}</span>
            {hint && (
              <Tooltip title={hint}>
                <InfoCircleOutlined style={{ fontSize: 11, opacity: 0.5 }} />
              </Tooltip>
            )}
          </Space>
        }
        value={value}
        suffix={suffix}
        precision={precision}
        styles={{ content: { fontSize: 26, fontWeight: 700, color } }}
      />
    </Card>
  );
}

/** Row of StatCards that stacks to 2-up and then 1-up as the viewport narrows.
 *
 * Five tiles can't use AntD's 24-column grid at all — 24/5 is 4.8, and a
 * fractional span produces a class name that doesn't exist, so the row
 * silently collapses. Above `xl` the five-tile case therefore switches to flex
 * basis, which divides evenly at any count; below it, both cases fall back to
 * the ordinary responsive spans. */
export function StatRow({ children, cols = 4 }: { children: ReactNode[]; cols?: 4 | 5 }) {
  const screens = Grid.useBreakpoint();
  const evenSplit = cols === 5 && screens.xl;

  return (
    <Row gutter={[16, 16]} style={{ marginBottom: 20 }} wrap={!evenSplit}>
      {children.map((child, i) => (
        <Col
          key={i}
          {...(evenSplit
            ? { flex: "1 1 0", style: { minWidth: 0 } }
            : { xs: 24, sm: 12, xl: 6 })}
        >
          {child}
        </Col>
      ))}
    </Row>
  );
}

/** Small muted caption under a chart or table — the "how to read this / what's
 *  the caveat" line. */
export function Caption({ children }: { children: ReactNode }) {
  return (
    <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 10, lineHeight: 1.6 }}>
      {children}
    </Text>
  );
}

/** recharts types a Tooltip `formatter`'s first argument as the full ValueType
 *  union (number | string | array) even on charts that only ever carry
 *  numbers, so a plain `(v: number) => …` no longer type-checks in recharts
 *  3.x. Taking `unknown` and coercing once here keeps every call site a
 *  one-liner instead of repeating the widening cast on every chart. */
export function tooltipValue(
  format: (n: number) => string,
  label?: string
): (v: unknown, name?: unknown) => [string, string] {
  return (v, name) => [format(Number(v)), label ?? String(name ?? "")];
}

/**
 * Renders the 784 raw pixel values the model actually sees.
 *
 * Drawn at native 28x28 into an offscreen ImageData and then scaled up by the
 * CSS box with `image-rendering: pixelated` — scaling the canvas backing store
 * instead would resample and blur, hiding exactly the detail (a collar, a
 * heel) that explains why a prediction went the way it did.
 */
export function PixelImage({
  pixels,
  size = 224,
  alt = "28 by 28 greyscale input",
}: {
  pixels: number[];
  size?: number;
  alt?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(255, Math.round(pixels[i] ?? 0)));
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      className="pixel-art"
      aria-label={alt}
      role="img"
      style={{
        width: "100%",
        maxWidth: size,
        aspectRatio: "1 / 1",
        height: "auto",
        display: "block",
        borderRadius: 6,
        border: "1px solid var(--color-border)",
        background: "#000",
      }}
    />
  );
}

/** A matplotlib PNG from the training notebook, on its own white plate so it
 *  stays legible in dark mode, and scrollable so a wide figure never widens
 *  the page. */
export function FigureCard({
  title,
  src,
  caption,
  minWidth = 560,
}: {
  title: string;
  src: string;
  caption?: ReactNode;
  minWidth?: number;
}) {
  return (
    <Card title={title} size="small" style={{ marginBottom: 20 }}>
      <div className="scroll-x">
        <div className="figure-plate" style={{ minWidth }}>
          <img src={src} alt={title} loading="lazy" style={{ width: "100%", display: "block" }} />
        </div>
      </div>
      {caption && <Caption>{caption}</Caption>}
    </Card>
  );
}

/** Standard loading / error / empty handling for a page that reads one
 *  endpoint, so seven pages don't each invent their own. Errors are shown with
 *  a retry rather than a dead end: on this deployment the most common cause is
 *  a backend that went back to sleep, and one retry usually fixes it. */
export function AsyncSection<T>({
  state,
  children,
  minHeight = 200,
}: {
  state: { data: T | null; error: string | null; loading: boolean; reload: () => void };
  children: (data: T) => ReactNode;
  minHeight?: number;
}) {
  if (state.loading && state.data === null) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight }}>
        <AntSpin />
      </div>
    );
  }
  if (state.error && state.data === null) {
    return (
      <AntResult
        status="warning"
        title="Could not load this section"
        subTitle={state.error}
        extra={
          <AntButton type="primary" onClick={state.reload}>
            Try again
          </AntButton>
        }
      />
    );
  }
  if (state.data === null) return null;
  return <>{children(state.data)}</>;
}
