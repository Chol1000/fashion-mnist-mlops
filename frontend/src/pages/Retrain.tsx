import { useCallback, useEffect, useRef, useState } from "react";
import {
  Card, Col, Row, Button, Upload, Space, Typography, Alert, Slider, Select, Checkbox,
  Steps, Table, Tag, Progress, Statistic, App as AntApp, Popconfirm, Empty,
} from "antd";
import {
  InboxOutlined, DatabaseOutlined, DownloadOutlined, DeleteOutlined, PlayCircleOutlined,
  CheckCircleFilled, LoadingOutlined,
} from "@ant-design/icons";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, Legend,
} from "recharts";
import { api, ApiError } from "../api";
import { useBackend } from "../context/BackendContext";
import { useAsync, pct } from "../hooks";
import { CLASS_NAMES, type RetrainStatus, type RetrainHistoryRow } from "../types";
import { PageHeader, Caption, useChartTheme, tooltipValue } from "../ui";
import WhatYouCanUpload from "../components/WhatYouCanUpload";

const { Text, Paragraph } = Typography;
const { Dragger } = Upload;

const PREPROCESS_STEPS = [
  ["Normalise", "Pixels 0–255 rescaled to 0.0–1.0 float32"],
  ["Resize", "28x28 greyscale expanded to 128x128 RGB for MobileNetV2"],
  ["Augment", "Random flip, brightness, contrast"],
  ["Split", "80% fine-tune / 20% validation"],
];

/** Parsed client-side purely to show the user what they're about to send. The
 *  API re-validates and cleans everything independently — this preview is not
 *  a gate. */
type CsvPreview = {
  rows: number;
  cols: number;
  head: Record<string, string>[];
  headCols: string[];
  distribution: { label: number; count: number }[];
  warning?: string;
};

function parseCsvPreview(text: string): CsvPreview | { error: string } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return { error: "The file has no data rows." };
  const header = lines[0].split(",").map((h) => h.trim());
  const labelIdx = header.indexOf("label");
  if (labelIdx === -1) return { error: "No `label` column found — the API requires one." };

  const pixelCount = header.filter((h) => /^pixel\d+$/.test(h)).length;
  const dist = new Map<number, number>();
  for (const line of lines.slice(1)) {
    const v = Number(line.split(",")[labelIdx]);
    if (Number.isInteger(v) && v >= 0 && v < 10) dist.set(v, (dist.get(v) ?? 0) + 1);
  }

  // First five pixel columns only — 785 columns of preview would be unreadable
  // and would dwarf the rest of the page.
  const headCols = ["label", ...header.filter((h) => /^pixel\d+$/.test(h)).slice(0, 5)];
  const idxs = headCols.map((c) => header.indexOf(c));
  const head = lines.slice(1, 9).map((line, i) => {
    const cells = line.split(",");
    const row: Record<string, string> = { key: String(i) };
    headCols.forEach((c, j) => (row[c] = cells[idxs[j]] ?? ""));
    return row;
  });

  return {
    rows: lines.length - 1,
    cols: header.length,
    head,
    headCols,
    distribution: [...dist.entries()].sort((a, b) => a[0] - b[0]).map(([label, count]) => ({ label, count })),
    warning:
      pixelCount !== 784
        ? `Found ${pixelCount} pixel columns; the API expects pixel1…pixel784 and will reject the file otherwise.`
        : undefined,
  };
}

function LiveProgress({ status }: { status: RetrainStatus }) {
  const chart = useChartTheme();
  const { phase, current_epoch, total_epochs, epoch_logs, steps, current_step, elapsed_sec } = status;

  // The backend reports a phase, not a percentage. Data loading and
  // preprocessing are quick and unmeasurable, so they get a fixed 10%; the
  // epoch counter drives the rest, which is the only part with real progress
  // to report.
  const percent =
    phase === "done"
      ? 100
      : phase === "training" && total_epochs > 0
        ? 10 + Math.min(Math.round((current_epoch / total_epochs) * 85), 85)
        : phase === "loading data" || phase === "preprocessing"
          ? 8
          : 96;

  const hasVal = epoch_logs.some((e) => e.val_accuracy !== null);

  return (
    <Card size="small" title="Retraining in progress" style={{ marginTop: 16 }}>
      <Progress percent={percent} status={status.running ? "active" : "success"} />
      <Text type="secondary" style={{ fontSize: 12 }}>
        Phase: {phase || "starting"} · {Math.round(elapsed_sec ?? 0)}s elapsed
        {total_epochs > 0 && ` · epoch ${current_epoch}/${total_epochs}`}
      </Text>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={11}>
          <Text strong style={{ fontSize: 13 }}>
            Pipeline steps
          </Text>
          <div style={{ marginTop: 8, fontSize: 12.5, lineHeight: 1.9 }}>
            {steps.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <CheckCircleFilled style={{ color: "#52c41a", marginTop: 4, flexShrink: 0 }} />
                <span>
                  {s.msg} <Text type="secondary" style={{ fontSize: 11 }}>({s.elapsed}s)</Text>
                </span>
              </div>
            ))}
            {current_step && steps[steps.length - 1]?.msg !== current_step && (
              <div className="step-active" style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <LoadingOutlined style={{ marginTop: 4, flexShrink: 0 }} />
                <Text type="secondary" italic>
                  {current_step}
                </Text>
              </div>
            )}
          </div>
        </Col>
        <Col xs={24} lg={13}>
          <Text strong style={{ fontSize: 13 }}>
            Accuracy per epoch
          </Text>
          {epoch_logs.length === 0 ? (
            <div style={{ height: 200, display: "flex", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Waiting for the first epoch to finish…
              </Text>
            </div>
          ) : (
            <div style={{ width: "100%", height: 200, marginTop: 8 }}>
              <ResponsiveContainer>
                <LineChart data={epoch_logs} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke={chart.grid} />
                  <XAxis dataKey="epoch" stroke={chart.axis} tick={{ fontSize: 11 }} />
                  <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} domain={["auto", 100]} />
                  <RTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => `${n.toFixed(2)}%`)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="accuracy" name="Train" stroke={chart.accent} strokeWidth={2} dot={{ r: 3 }} />
                  {hasVal && (
                    <Line
                      dataKey="val_accuracy"
                      name="Validation"
                      stroke={chart.accent2}
                      strokeWidth={2}
                      strokeDasharray="5 4"
                      dot={{ r: 3 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Col>
      </Row>
    </Card>
  );
}

export default function Retrain() {
  const { message } = AntApp.useApp();
  const { health, refresh: refreshHealth } = useBackend();
  const dbSamples = health?.db_samples ?? 0;

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | { error: string } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [sampleSize, setSampleSize] = useState(300);

  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(64);
  const [clearAfter, setClearAfter] = useState(false);
  const [starting, setStarting] = useState(false);
  const [status, setStatus] = useState<RetrainStatus | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  const history = useAsync(() => api.retrainHistory(), []);
  const chart = useChartTheme();

  // Poll while a run is live. The interval is cleared by the effect's own
  // cleanup on unmount, so navigating away mid-run doesn't leave a request
  // loop behind — the run itself continues server-side and is picked up again
  // by the resume probe below.
  const polling = useRef(false);
  const poll = useCallback(async () => {
    if (polling.current) return;
    polling.current = true;
    try {
      const s = await api.retrainStatus();
      setStatus(s);
      if (!s.running) {
        refreshHealth();
        history.reload();
      }
    } catch {
      /* a transient failure mid-training is expected — TensorFlow can starve
         the single worker on a free CPU Space. Keep the last status on screen
         and try again on the next tick. */
    } finally {
      polling.current = false;
    }
  }, [refreshHealth, history]);

  useEffect(() => {
    // One probe on mount so a run started before a reload (or from another
    // tab) is picked up rather than looking like nothing is happening.
    poll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!status?.running) return;
    const t = setInterval(poll, 1500);
    return () => clearInterval(t);
  }, [status?.running, poll]);

  const onFile = (f: File) => {
    setFile(f);
    setPreview(null);
    f.text().then((t) => setPreview(parseCsvPreview(t))).catch(() => setPreview({ error: "Could not read the file." }));
    return false;
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const res = await api.uploadData(file);
      message.success(`${res.samples_added} samples stored — ${res.total_in_db} now in the database.`);
      setFile(null);
      setPreview(null);
      refreshHealth();
    } catch (e) {
      message.error(e instanceof ApiError ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const clearDb = async () => {
    try {
      await api.clearUploaded();
      message.success("Uploaded samples cleared.");
      refreshHealth();
    } catch (e) {
      message.error(e instanceof ApiError ? e.message : String(e));
    }
  };

  const start = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const res = await api.startRetrain(epochs, batchSize, clearAfter);
      message.success(res.message);
      // Seed a running status immediately so the progress card appears without
      // waiting for the first poll — a two-second dead gap after clicking
      // reads as a button that did nothing.
      setStatus({
        running: true, elapsed_sec: 0, phase: "loading data", current_epoch: 0,
        total_epochs: epochs, epoch_logs: [], n_samples: res.samples, steps: [],
        current_step: "Starting", last_result: null,
      });
      poll();
    } catch (e) {
      setStartError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const stage = dbSamples > 0 ? (status?.last_result ? 2 : 2) : 0;
  const last = status?.last_result;
  const finished = !status?.running && last && !last.error;

  const historyRows: RetrainHistoryRow[] = history.data?.history ?? [];

  return (
    <>
      <PageHeader
        eyebrow="Continuous training"
        title="Upload data & retrain"
        subtitle="The full retraining loop: labelled rows go into SQLite, get preprocessed, and fine-tune the saved MobileNetV2 checkpoint. Every stage below is a real call against the running API."
      />

      <Card size="small" style={{ marginBottom: 20 }}>
        <Steps
          size="small"
          current={stage}
          responsive
          items={[
            { title: "Upload", description: "Labelled CSV into SQLite" },
            { title: "Preprocess", description: "Normalise, resize, augment" },
            { title: "Fine-tune", description: "Adam lr=1e-4, early stopping" },
          ]}
        />
      </Card>

      {/* ── Stage 1 ────────────────────────────────────────────────────────── */}
      <Card
        size="small"
        title="1 · Upload training data"
        style={{ marginBottom: 20 }}
        extra={
          <Space>
            <Tag icon={<DatabaseOutlined />} color={dbSamples > 0 ? "green" : "default"}>
              {dbSamples} sample{dbSamples === 1 ? "" : "s"} in database
            </Tag>
            {dbSamples > 0 && (
              <Popconfirm
                title="Clear all uploaded samples?"
                description="This deletes every row you have uploaded. The trained model is not affected."
                okText="Clear"
                okButtonProps={{ danger: true }}
                onConfirm={clearDb}
              >
                <Button size="small" danger icon={<DeleteOutlined />}>
                  Clear
                </Button>
              </Popconfirm>
            )}
          </Space>
        }
      >
        <Row gutter={[20, 20]}>
          <Col xs={24} lg={14}>
            <div style={{ marginBottom: 14 }}>
              <WhatYouCanUpload variant="csv" />
            </div>
            <Dragger accept=".csv" maxCount={1} showUploadList={false} beforeUpload={onFile}>
              <p className="ant-upload-drag-icon" style={{ marginBottom: 4 }}>
                <InboxOutlined />
              </p>
              <p className="ant-upload-text" style={{ fontSize: 14 }}>
                Drop a labelled CSV, or click to browse
              </p>
              <p className="ant-upload-hint" style={{ fontSize: 12 }}>
                Columns: <code>label</code> (0–9) plus <code>pixel1</code>…<code>pixel784</code> (0–255).
              </p>
            </Dragger>

            {preview && "error" in preview && (
              <Alert type="error" showIcon message={preview.error} style={{ marginTop: 12 }} />
            )}

            {preview && !("error" in preview) && (
              <div style={{ marginTop: 12 }}>
                <Alert
                  type="success"
                  showIcon
                  message={`${preview.rows} rows · ${preview.cols} columns (1 label + ${preview.cols - 1} pixel features)`}
                  style={{ marginBottom: 10 }}
                />
                {preview.warning && (
                  <Alert type="warning" showIcon message={preview.warning} style={{ marginBottom: 10 }} />
                )}
                <Text type="secondary" style={{ fontSize: 12 }}>
                  First 8 rows, first 5 of the pixel columns:
                </Text>
                <Table
                  size="small"
                  style={{ marginTop: 6 }}
                  dataSource={preview.head}
                  pagination={false}
                  scroll={{ x: true }}
                  columns={preview.headCols.map((c) => ({
                    title: c === "label" ? "Label" : c,
                    dataIndex: c,
                    key: c,
                    render: (v: string) =>
                      c === "label" ? (
                        <Tag>{CLASS_NAMES[Number(v)] ?? v}</Tag>
                      ) : (
                        <span className="tabular">{v}</span>
                      ),
                  }))}
                />
                <div style={{ marginTop: 10 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Class distribution:{" "}
                  </Text>
                  <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
                    {preview.distribution.map((d) => (
                      <Tag key={d.label} style={{ marginInlineEnd: 0 }}>
                        {CLASS_NAMES[d.label]}: {d.count}
                      </Tag>
                    ))}
                  </Space>
                </div>
                <Button
                  type="primary"
                  loading={uploading}
                  onClick={upload}
                  style={{ marginTop: 14 }}
                  disabled={!!preview.warning}
                >
                  Upload to database
                </Button>
              </div>
            )}
          </Col>

          <Col xs={24} lg={10}>
            <Card size="small" type="inner" title="No CSV to hand?">
              <Paragraph type="secondary" style={{ fontSize: 13, lineHeight: 1.7 }}>
                Download a correctly-shaped slice of the Fashion MNIST training set to upload straight
                back — enough to exercise the whole loop without hunting for data.
              </Paragraph>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Rows: <Text strong style={{ fontSize: 12 }}>{sampleSize}</Text>
              </Text>
              <Slider min={60} max={3000} step={20} value={sampleSize} onChange={setSampleSize} />
              <Button
                icon={<DownloadOutlined />}
                href={api.sampleCsvUrl(sampleSize)}
                // The browser downloads straight from the API rather than
                // routing several megabytes of CSV through this page.
                download
                block
              >
                Download {sampleSize}-row CSV
              </Button>
              <Caption>
                Fine-tuning on data the model already trained on will not teach it anything new — it is
                here to demonstrate the pipeline, not to improve accuracy.
              </Caption>
            </Card>
          </Col>
        </Row>
      </Card>

      {/* ── Stage 2 ────────────────────────────────────────────────────────── */}
      <Card size="small" title="2 · Preprocessing" style={{ marginBottom: 20 }}>
        <Paragraph type="secondary" style={{ fontSize: 13, marginTop: 0, lineHeight: 1.7 }}>
          Applied automatically by <code>FashionMNISTPreprocessor</code> to every uploaded row before
          training starts — the same transforms used for the original training run, so fine-tuned
          weights stay compatible with what the model already learned.
        </Paragraph>
        <Row gutter={[12, 12]}>
          {PREPROCESS_STEPS.map(([title, desc]) => (
            <Col key={title} xs={12} lg={6}>
              <Card size="small" type="inner" styles={{ body: { padding: "12px 14px" } }}>
                <Text strong style={{ fontSize: 13 }}>
                  {title}
                </Text>
                <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 4, lineHeight: 1.6 }}>
                  {desc}
                </Text>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* ── Stage 3 ────────────────────────────────────────────────────────── */}
      <Card size="small" title="3 · Fine-tune the model" style={{ marginBottom: 20 }}>
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="What this actually does"
          description={
            <span>
              Loads <code>fashion_model.h5</code> — the MobileNetV2 checkpoint at 93.17% test accuracy —
              and fine-tunes its final layers on the uploaded samples with Adam at lr=1e-4 and early
              stopping. The running API reloads the new weights the moment training finishes, so the
              Classify page starts using them immediately.
            </span>
          }
        />

        <Row gutter={[16, 16]} align="bottom">
          <Col xs={24} md={10}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Max epochs: <Text strong style={{ fontSize: 12 }}>{epochs}</Text>
            </Text>
            <Slider min={1} max={20} value={epochs} onChange={setEpochs} disabled={status?.running} />
          </Col>
          <Col xs={12} md={6}>
            <Text type="secondary" style={{ fontSize: 12, display: "block", marginBottom: 4 }}>
              Batch size
            </Text>
            <Select
              value={batchSize}
              onChange={setBatchSize}
              disabled={status?.running}
              style={{ width: "100%" }}
              options={[32, 64, 128].map((v) => ({ value: v, label: String(v) }))}
            />
          </Col>
          <Col xs={12} md={8}>
            <Checkbox
              checked={clearAfter}
              onChange={(e) => setClearAfter(e.target.checked)}
              disabled={status?.running}
            >
              <Text style={{ fontSize: 13 }}>Clear database afterwards</Text>
            </Checkbox>
          </Col>
        </Row>

        <Button
          type="primary"
          icon={<PlayCircleOutlined />}
          size="large"
          loading={starting || status?.running}
          disabled={dbSamples === 0}
          onClick={start}
          style={{ marginTop: 18 }}
        >
          {status?.running ? "Retraining…" : "Start retraining"}
        </Button>
        {dbSamples === 0 && (
          <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 8 }}>
            Upload some labelled samples first — there is nothing to fine-tune on yet.
          </Text>
        )}

        {startError && (
          <Alert type="error" showIcon message="Could not start" description={startError} style={{ marginTop: 14 }} />
        )}

        {status && (status.running || status.steps.length > 0) && <LiveProgress status={status} />}

        {finished && last && (
          <Card size="small" title="Run complete" style={{ marginTop: 16 }}>
            <Row gutter={[16, 16]}>
              {(
                [
                  ["Accuracy", last.accuracy !== undefined ? pct(last.accuracy) : "—"],
                  ["Macro F1", last.f1_score?.toFixed(4) ?? "—"],
                  ["Precision", last.precision?.toFixed(4) ?? "—"],
                  ["Recall", last.recall?.toFixed(4) ?? "—"],
                  ["Epochs ran", String(last.epochs_ran ?? "—")],
                ] as [string, string][]
              ).map(([label, value]) => (
                <Col key={label} xs={12} md={8} xl={4}>
                  <Statistic
                    title={<span style={{ fontSize: 12 }}>{label}</span>}
                    value={value}
                    styles={{ content: { fontSize: 22, fontWeight: 700 } }}
                  />
                </Col>
              ))}
            </Row>
            <Caption>
              These are measured on the validation split carved out of your uploaded samples — a much
              smaller and different set from the original 10,000-image test split, so they are not
              directly comparable to the baseline on Model Metrics.
            </Caption>
          </Card>
        )}

        {!status?.running && last?.error && (
          <Alert type="error" showIcon message="Last run failed" description={last.error} style={{ marginTop: 16 }} />
        )}
      </Card>

      {/* ── History ────────────────────────────────────────────────────────── */}
      <Card size="small" title="Retraining history" extra={<Button size="small" onClick={history.reload}>Refresh</Button>}>
        {historyRows.length === 0 ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="No retraining runs recorded yet." />
        ) : (
          <>
            <Table
              size="small"
              rowKey={(r) => String(r.id ?? r.trained_at)}
              dataSource={historyRows}
              scroll={{ x: true }}
              pagination={historyRows.length > 10 ? { pageSize: 10 } : false}
              columns={[
                { title: "When", dataIndex: "trained_at", key: "trained_at" },
                { title: "Samples", dataIndex: "samples_used", key: "samples_used", align: "right" },
                {
                  title: "Accuracy",
                  dataIndex: "accuracy",
                  key: "accuracy",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v != null ? pct(v) : "—"}</span>,
                },
                {
                  title: "F1",
                  dataIndex: "f1_score",
                  key: "f1_score",
                  align: "right",
                  render: (v: number) => <span className="tabular">{v?.toFixed(4) ?? "—"}</span>,
                },
                { title: "Epochs", dataIndex: "epochs_ran", key: "epochs_ran", align: "right" },
                { title: "Notes", dataIndex: "notes", key: "notes" },
              ]}
            />
            {historyRows.length > 1 && (
              <div style={{ width: "100%", height: 220, marginTop: 16 }}>
                <ResponsiveContainer>
                  <LineChart
                    data={[...historyRows].reverse().map((r, i) => ({ run: i + 1, accuracy: (r.accuracy ?? 0) * 100 }))}
                    margin={{ top: 8, right: 8, left: -12, bottom: 0 }}
                  >
                    <CartesianGrid stroke={chart.grid} />
                    <XAxis dataKey="run" stroke={chart.axis} tick={{ fontSize: 11 }} />
                    <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} />
                    <RTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => `${n.toFixed(2)}%`, "Accuracy")} />
                    <Line dataKey="accuracy" stroke={chart.accent} strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}
      </Card>
    </>
  );
}
