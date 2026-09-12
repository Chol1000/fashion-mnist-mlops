import { Card, Col, Row, Table, Tag, Typography, Space, Button, Alert, Badge, Descriptions } from "antd";
import {
  ReloadOutlined, ApiOutlined, CheckCircleOutlined, DatabaseOutlined, ClockCircleOutlined,
  CloudServerOutlined,
} from "@ant-design/icons";
import { api, API_BASE, API_IS_REMOTE } from "../api";
import { SPACE_URL } from "../config";
import { useBackend } from "../context/BackendContext";
import { useAsync, formatUptime, pct } from "../hooks";
import { PageHeader, StatCard, StatRow, Caption } from "../ui";

const { Text, Paragraph } = Typography;

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/", "Service info and uptime"],
  ["GET", "/health", "Model readiness and database stats"],
  ["POST", "/predict", "Classify from 784 pixel values"],
  ["POST", "/predict/image", "Classify from PNG, JPG, WebP or AVIF"],
  ["GET", "/sample/random", "Random held-out test image, optionally filtered by class"],
  ["GET", "/sample/csv", "Download a correctly-shaped slice of the training set"],
  ["POST", "/upload-data", "Store a labelled CSV in SQLite"],
  ["POST", "/retrain", "Start background fine-tuning"],
  ["GET", "/retrain/status", "Poll retraining progress"],
  ["GET", "/retrain/history", "Past retraining runs"],
  ["DELETE", "/uploaded-data", "Clear stored samples"],
  ["GET", "/metrics", "Saved evaluation metrics"],
  ["GET", "/insights", "Dataset statistics"],
  ["GET", "/docs", "Swagger UI"],
];

const METHOD_COLOR: Record<string, string> = {
  GET: "blue",
  POST: "green",
  DELETE: "red",
};

export default function System() {
  const { health, online, refresh } = useBackend();
  const retrain = useAsync(() => api.retrainStatus(), []);

  const last = retrain.data?.last_result;

  return (
    <>
      <PageHeader
        eyebrow="Operations"
        title="API & system status"
        subtitle="Live service health, the state of any retraining run, and the full endpoint reference."
        extra={
          <Space wrap>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                refresh();
                retrain.reload();
              }}
            >
              Refresh
            </Button>
            <Button type="primary" icon={<ApiOutlined />} href={api.docsUrl()} target="_blank" rel="noreferrer">
              Swagger UI
            </Button>
          </Space>
        }
      />

      <StatRow>
        <StatCard
          label="Service"
          value={online ? "Online" : "Unreachable"}
          icon={<CheckCircleOutlined />}
          color={online ? undefined : "#cf1322"}
        />
        <StatCard
          label="Model"
          value={
            health?.model_ready
              ? "Loaded"
              : health?.model_file === false
                ? "Missing"
                : "Failed to load"
          }
          icon={<CloudServerOutlined />}
          color={health?.model_ready ? undefined : "#cf1322"}
          hint={
            health?.model_ready
              ? "The MobileNetV2 weights are in memory and can serve a prediction right now."
              : health?.model_file === false
                ? "No checkpoint at models/fashion_model.h5 — it was never shipped into this image."
                : "The checkpoint is on disk but TensorFlow could not load it. Check the container logs."
          }
        />
        <StatCard
          label="Uptime"
          value={formatUptime(health?.uptime_sec ?? 0)}
          icon={<ClockCircleOutlined />}
          hint="Since the container last started."
        />
        <StatCard
          label="Samples in database"
          value={health?.db_samples ?? 0}
          icon={<DatabaseOutlined />}
          hint="Uploaded rows available for the next fine-tuning run."
        />
      </StatRow>

      <Card size="small" title="Deployment" style={{ marginBottom: 20 }}>
        <Descriptions
          size="small"
          column={{ xs: 1, lg: 2 }}
          items={[
            {
              key: "base",
              label: "API origin",
              children: <code style={{ fontSize: 12 }}>{API_BASE || "same origin as this page"}</code>,
            },
            {
              key: "shape",
              label: "Topology",
              children: API_IS_REMOTE
                ? "Dashboard and API deployed separately"
                : "Single container — FastAPI serves both this dashboard and the API",
            },
            {
              key: "space",
              label: "Space",
              children: (
                <a href={SPACE_URL} target="_blank" rel="noreferrer">
                  {SPACE_URL.replace("https://huggingface.co/spaces/", "")}
                </a>
              ),
            },
            {
              key: "sleep",
              label: "Sleep policy",
              children: "Free CPU Space — sleeps after 48h without traffic, wakes on the next request",
            },
          ]}
        />
        {API_IS_REMOTE && (
          <Alert
            type="warning"
            showIcon
            style={{ marginTop: 14 }}
            message="Two independently-sleeping services"
            description="Because the dashboard and the API are separate Spaces, opening this page does not by itself keep the API awake — it only wakes it on demand, which costs a cold start. Serving both from one container removes the problem entirely."
          />
        )}
      </Card>

      <Card
        size="small"
        title="Retraining status"
        style={{ marginBottom: 20 }}
        extra={
          retrain.data?.running ? (
            <Badge status="processing" text="Running" />
          ) : (
            <Badge status="default" text="Idle" />
          )
        }
      >
        {retrain.error && <Alert type="warning" showIcon message={retrain.error} />}
        {retrain.data?.running && (
          <Alert
            type="info"
            showIcon
            message={`In progress — phase: ${retrain.data.phase}, ${Math.round(retrain.data.elapsed_sec ?? 0)}s elapsed`}
            description={
              retrain.data.total_epochs > 0
                ? `Epoch ${retrain.data.current_epoch} of ${retrain.data.total_epochs} on ${retrain.data.n_samples} samples.`
                : undefined
            }
          />
        )}
        {!retrain.data?.running && last && !last.error && (
          <Row gutter={[16, 16]}>
            {(
              [
                ["Accuracy", last.accuracy != null ? pct(last.accuracy) : "—"],
                ["Macro F1", last.f1_score?.toFixed(4) ?? "—"],
                ["Precision", last.precision?.toFixed(4) ?? "—"],
                ["Recall", last.recall?.toFixed(4) ?? "—"],
                ["Epochs ran", String(last.epochs_ran ?? "—")],
                ["Samples", String(last.samples ?? "—")],
              ] as [string, string][]
            ).map(([label, value]) => (
              <Col key={label} xs={12} md={8} xl={4}>
                <Text type="secondary" style={{ fontSize: 12, display: "block" }}>
                  {label}
                </Text>
                <Text strong className="tabular" style={{ fontSize: 18 }}>
                  {value}
                </Text>
              </Col>
            ))}
          </Row>
        )}
        {!retrain.data?.running && last?.error && (
          <Alert type="error" showIcon message="Last run failed" description={last.error} />
        )}
        {!retrain.data?.running && !last && (
          <Paragraph type="secondary" style={{ fontSize: 13, margin: 0 }}>
            No retraining has run since the container started.
          </Paragraph>
        )}
      </Card>

      <Card size="small" title="API endpoints">
        <Table
          size="small"
          rowKey={(r) => r[0] + r[1]}
          dataSource={ENDPOINTS}
          pagination={false}
          scroll={{ x: true }}
          columns={[
            {
              title: "Method",
              key: "method",
              width: 90,
              render: (_, r) => (
                <Tag color={METHOD_COLOR[r[0]]} style={{ marginInlineEnd: 0, fontWeight: 600 }}>
                  {r[0]}
                </Tag>
              ),
            },
            {
              title: "Path",
              key: "path",
              render: (_, r) => <code style={{ fontSize: 12.5 }}>{r[1]}</code>,
            },
            { title: "Description", key: "desc", render: (_, r) => r[2] },
          ]}
        />
        <Caption>
          Paths are served at the root of the API origin above, with no <code>/api</code> prefix — the
          same paths the Locust load tests and the README examples use.
        </Caption>
      </Card>
    </>
  );
}
