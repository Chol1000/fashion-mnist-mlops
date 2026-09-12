import { useCallback, useState } from "react";
import {
  Card, Col, Row, Button, Segmented, Select, Upload, Space, Typography, Alert, Input, Spin,
} from "antd";
import {
  InboxOutlined, ThunderboltOutlined, ReloadOutlined, PictureOutlined,
} from "@ant-design/icons";
import { api, ApiError } from "../api";
import { CLASS_NAMES, type PredictResult, type RandomSample } from "../types";
import { PageHeader, PixelImage } from "../ui";
import PredictionResult from "../components/PredictionResult";
import DrawPad from "../components/DrawPad";
import WhatYouCanUpload from "../components/WhatYouCanUpload";

const { Text, Paragraph } = Typography;
const { Dragger } = Upload;

type Method = "Upload image" | "Test sample" | "Draw" | "Paste pixels";
const METHODS: Method[] = ["Upload image", "Test sample", "Draw", "Paste pixels"];

export default function Predict() {
  const [method, setMethod] = useState<Method>("Upload image");

  // Result state is shared across methods but cleared on every switch: leaving
  // the previous method's prediction on screen next to a new input is the
  // single most confusing thing this page could do.
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [sample, setSample] = useState<RandomSample | null>(null);
  const [classFilter, setClassFilter] = useState<number | undefined>(undefined);
  const [drawn, setDrawn] = useState<number[] | null>(null);
  const [pasted, setPasted] = useState("");

  const reset = (next: Method) => {
    setMethod(next);
    setResult(null);
    setError(null);
  };

  const run = useCallback(async (fn: () => Promise<PredictResult>) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await fn());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const loadSample = useCallback(async () => {
    setError(null);
    setResult(null);
    try {
      setSample(await api.randomSample(classFilter));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, [classFilter]);

  // Tagged rather than a bare `{error} | {pixels}` union: the render below
  // branches on it three times, and a discriminant narrows cleanly where an
  // `in` check does not.
  type Parsed =
    | { ok: true; pixels: number[] }
    | { ok: false; error: string };

  const parsedPixels: Parsed | null = (() => {
    if (!pasted.trim()) return null;
    const parts = pasted.split(/[,\s]+/).filter(Boolean).map(Number);
    if (parts.some(Number.isNaN))
      return { ok: false, error: "Values must be numbers separated by commas or spaces." };
    if (parts.length !== 784)
      return { ok: false, error: `Expected 784 values, found ${parts.length}.` };
    return { ok: true, pixels: parts };
  })();

  return (
    <>
      <PageHeader
        eyebrow="Inference"
        title="Classify a garment"
        subtitle="Four ways in, one model behind them. Everything on this page is a live call to POST /predict or POST /predict/image on the running API."
      />

      <Segmented
        options={METHODS}
        value={method}
        onChange={(v) => reset(v as Method)}
        style={{ marginBottom: 20 }}
      />

      <Row gutter={[16, 16]} align="top">
        <Col xs={24} xl={9}>
          <Card size="small" title="Input">
            {method === "Upload image" && (
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                <Dragger
                  accept=".png,.jpg,.jpeg,.webp,.avif"
                  maxCount={1}
                  showUploadList={false}
                  // The API does the real work; uploading here would double the
                  // request. `beforeUpload` returning false keeps AntD's
                  // drag-and-drop affordance without its uploader.
                  beforeUpload={(f) => {
                    setFile(f);
                    setResult(null);
                    setError(null);
                    setFilePreview((prev) => {
                      if (prev) URL.revokeObjectURL(prev);
                      return URL.createObjectURL(f);
                    });
                    return false;
                  }}
                >
                  <p className="ant-upload-drag-icon" style={{ marginBottom: 4 }}>
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text" style={{ fontSize: 14 }}>
                    Drop a clothing photo, or click to browse
                  </p>
                  <p className="ant-upload-hint" style={{ fontSize: 12 }}>
                    PNG, JPG, WebP or AVIF, any size — the API converts it to 28x28 greyscale.
                  </p>
                </Dragger>

                {file && filePreview && (
                  <>
                    <img
                      src={filePreview}
                      alt={file.name}
                      style={{
                        width: "100%",
                        maxWidth: 224,
                        borderRadius: 6,
                        border: "1px solid var(--color-border)",
                        display: "block",
                      }}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {file.name}
                    </Text>
                    <Button
                      type="primary"
                      icon={<ThunderboltOutlined />}
                      loading={busy}
                      onClick={() => run(() => api.predictImage(file))}
                      block
                    >
                      Classify image
                    </Button>
                  </>
                )}

              </Space>
            )}

            {method === "Test sample" && (
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                <Paragraph type="secondary" style={{ fontSize: 13, marginBottom: 0, lineHeight: 1.7 }}>
                  Pulls a random image from the 10,000-image test split — data the model never saw
                  during training — so the prediction can be checked against its true label.
                </Paragraph>
                <Select
                  allowClear
                  placeholder="Any class"
                  value={classFilter}
                  onChange={(v) => setClassFilter(v)}
                  options={CLASS_NAMES.map((name, i) => ({ value: i, label: name }))}
                  style={{ width: "100%" }}
                />
                <Button icon={<ReloadOutlined />} onClick={loadSample} block>
                  Load a random sample
                </Button>

                {sample && (
                  <>
                    <PixelImage pixels={sample.pixels} />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      True label: <Text strong style={{ fontSize: 12 }}>{sample.label_name}</Text>
                    </Text>
                    <Button
                      type="primary"
                      icon={<ThunderboltOutlined />}
                      loading={busy}
                      onClick={() => run(() => api.predictPixels(sample.pixels))}
                      block
                    >
                      Classify
                    </Button>
                  </>
                )}
              </Space>
            )}

            {method === "Draw" && (
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                <DrawPad
                  onChange={(p) => {
                    setDrawn(p);
                    setResult(null);
                  }}
                />
                {drawn && (
                  <>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      What the model actually receives:
                    </Text>
                    <PixelImage pixels={drawn} size={120} alt="Downsampled drawing" />
                  </>
                )}
                <Button
                  type="primary"
                  icon={<ThunderboltOutlined />}
                  loading={busy}
                  disabled={!drawn}
                  onClick={() => drawn && run(() => api.predictPixels(drawn))}
                  block
                >
                  Classify drawing
                </Button>
              </Space>
            )}

            {method === "Paste pixels" && (
              <Space direction="vertical" size={14} style={{ width: "100%" }}>
                <Paragraph type="secondary" style={{ fontSize: 13, marginBottom: 0, lineHeight: 1.7 }}>
                  784 values in 0–255, comma or whitespace separated — one row of a Fashion MNIST CSV
                  with the label column removed. This is the raw shape <code>POST /predict</code>{" "}
                  expects.
                </Paragraph>
                <Input.TextArea
                  rows={6}
                  value={pasted}
                  onChange={(e) => {
                    setPasted(e.target.value);
                    setResult(null);
                  }}
                  placeholder="0, 0, 0, 12, 45, 200, …"
                  style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12 }}
                />
                {parsedPixels && !parsedPixels.ok && (
                  <Alert type="warning" showIcon message={parsedPixels.error} />
                )}
                {parsedPixels?.ok && <PixelImage pixels={parsedPixels.pixels} size={140} />}
                <Button
                  type="primary"
                  icon={<ThunderboltOutlined />}
                  loading={busy}
                  disabled={!parsedPixels?.ok}
                  onClick={() => parsedPixels?.ok && run(() => api.predictPixels(parsedPixels.pixels))}
                  block
                >
                  Classify
                </Button>
              </Space>
            )}
          </Card>
        </Col>

        <Col xs={24} xl={15}>
          {error && (
            <Alert type="error" showIcon message="Prediction failed" description={error} style={{ marginBottom: 16 }} />
          )}

          {busy && !result && (
            <Card size="small" style={{ minHeight: 300, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Spin tip="Running inference…">
                <div style={{ padding: 30 }} />
              </Spin>
            </Card>
          )}

          {result && (
            <PredictionResult
              result={result}
              trueLabel={method === "Test sample" ? sample?.label_name : undefined}
            />
          )}

          {!result && !busy && !error && (
            <Card
              size="small"
              title={
                <Space size={8}>
                  <PictureOutlined style={{ opacity: 0.45 }} />
                  <span>What this model can classify</span>
                </Space>
              }
              extra={
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Results appear here
                </Text>
              }
            >
              <WhatYouCanUpload variant={method === "Upload image" ? "image" : "plain"} bordered={false} />
            </Card>
          )}
        </Col>
      </Row>
    </>
  );
}
