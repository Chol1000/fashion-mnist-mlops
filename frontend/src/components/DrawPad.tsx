import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Slider, Space, Typography } from "antd";
import { ClearOutlined } from "@ant-design/icons";

const { Text } = Typography;

const CANVAS = 280; // 10x the model's input, so one drawn pixel maps cleanly

/**
 * Draw-your-own garment pad.
 *
 * Fashion MNIST images are light garments on a black ground, so the pad is
 * black with a white brush — drawing the inverse (dark on white, the reflex
 * from a signature pad) produces confident nonsense, which is why the hint
 * below spells the convention out.
 *
 * The visible canvas is 280x280 and is downsampled to 28x28 on export via a
 * second offscreen canvas: letting the browser's image smoothing average each
 * 10x10 block is what gives the soft edges the model was trained on. Drawing
 * directly at 28x28 gives hard aliased strokes and noticeably worse
 * predictions.
 */
export default function DrawPad({ onChange }: { onChange: (pixels: number[] | null) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const [brush, setBrush] = useState(22);
  const [dirty, setDirty] = useState(false);

  const clear = useCallback(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, CANVAS, CANVAS);
    setDirty(false);
    onChange(null);
  }, [onChange]);

  useEffect(() => {
    clear();
  }, [clear]);

  /** Export at 28x28 — the shape /predict expects. */
  const emit = useCallback(() => {
    const src = ref.current;
    if (!src) return;
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d");
    if (!sctx) return;
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(src, 0, 0, 28, 28);
    const { data } = sctx.getImageData(0, 0, 28, 28);
    const pixels: number[] = new Array(784);
    for (let i = 0; i < 784; i++) {
      // Already greyscale (white brush on black), so any channel is the value.
      pixels[i] = data[i * 4];
    }
    onChange(pixels);
  }, [onChange]);

  /** Pointer events rather than mouse+touch pairs: one code path covers mouse,
   *  trackpad, pen and finger, and setPointerCapture keeps a stroke alive when
   *  the pointer leaves the canvas mid-drag. */
  const pos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * CANVAS,
      y: ((e.clientY - rect.top) / rect.height) * CANVAS,
    };
  };

  const down = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    last.current = pos(e);
    stroke(e); // a tap with no movement should still leave a dot
  };

  const stroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const p = pos(e);
    const from = last.current ?? p;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
    setDirty(true);
  };

  const up = () => {
    if (!drawing.current) return;
    drawing.current = false;
    last.current = null;
    // Export on stroke end rather than on every pointer move: a full 28x28
    // resample per move event is wasted work no one sees.
    emit();
  };

  return (
    <Space direction="vertical" size={10} style={{ width: "100%" }}>
      <canvas
        ref={ref}
        width={CANVAS}
        height={CANVAS}
        onPointerDown={down}
        onPointerMove={stroke}
        onPointerUp={up}
        onPointerCancel={up}
        style={{
          width: "100%",
          maxWidth: CANVAS,
          aspectRatio: "1 / 1",
          height: "auto",
          display: "block",
          borderRadius: 8,
          border: "1px solid var(--color-border)",
          cursor: "crosshair",
          // Without this a finger drag scrolls the page instead of drawing.
          touchAction: "none",
        }}
      />
      <Space size={12} style={{ width: "100%", maxWidth: CANVAS }}>
        <Text type="secondary" style={{ fontSize: 12, whiteSpace: "nowrap" }}>
          Brush
        </Text>
        <Slider min={8} max={40} value={brush} onChange={setBrush} style={{ flex: 1, minWidth: 90 }} />
        <Button size="small" icon={<ClearOutlined />} onClick={clear} disabled={!dirty}>
          Clear
        </Button>
      </Space>
      <Text type="secondary" style={{ fontSize: 12, lineHeight: 1.6 }}>
        Draw a light garment silhouette on the black ground — that is the convention the model was
        trained on. A dark shape on a light ground will classify badly no matter how good the
        drawing is.
      </Text>
    </Space>
  );
}
