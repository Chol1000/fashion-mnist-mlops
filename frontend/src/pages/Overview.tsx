import { useNavigate } from "react-router-dom";
import { Card, Col, Row, Button, Space, Typography, Tag, Progress, Alert } from "antd";
import {
  AimOutlined, DatabaseOutlined, ClockCircleOutlined, ExperimentOutlined,
  CloudUploadOutlined, LineChartOutlined, ArrowRightOutlined, TrophyOutlined,
} from "@ant-design/icons";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  LineChart, Line, Legend, Cell,
} from "recharts";
import { api } from "../api";
import { useAsync, formatUptime, pct } from "../hooks";
import { useBackend } from "../context/BackendContext";
import {
  PageHeader, StatCard, StatRow, Caption, useChartTheme, useConfColors, tooltipValue, AsyncSection,
} from "../ui";
import type { Metrics } from "../types";

const { Text, Paragraph } = Typography;

/** Which evaluation block is "current": a fine-tuned model supersedes the
 *  original test-set numbers, so the headline KPIs follow the retrain when one
 *  has happened. */
function headline(m: Metrics) {
  const base = m.evaluation ?? m.initial_training;
  return { base, retrain: m.retrain, current: m.retrain ?? base };
}

function PerClassF1({ m }: { m: Metrics }) {
  const chart = useChartTheme();
  const conf = useConfColors();
  const f1 = (m.evaluation ?? m.initial_training)?.per_class_f1;
  if (!f1) return null;

  const rows = Object.entries(f1)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const mean = rows.reduce((s, r) => s + r.value, 0) / rows.length;

  return (
    <Card title="Per-class F1 — where the model is strong and weak" size="small">
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 56 }}>
            <CartesianGrid stroke={chart.grid} vertical={false} />
            <XAxis
              dataKey="name"
              stroke={chart.axis}
              tick={{ fontSize: 11 }}
              angle={-35}
              textAnchor="end"
              interval={0}
            />
            <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} domain={[0, 1]} />
            <RTooltip
              contentStyle={chart.tooltip}
              cursor={{ fill: chart.dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }}
              formatter={tooltipValue((n) => n.toFixed(4), "F1")}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={46}>
              {rows.map((r) => (
                // Coloured by the same thresholds the prediction cards use, so
                // "this class is weak" reads identically everywhere.
                <Cell key={r.name} fill={r.value >= 0.95 ? conf.high : r.value >= 0.85 ? chart.accent : conf.mid} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Caption>
        Macro mean {mean.toFixed(4)} across 10,000 held-out test images. Shirt is the weakest class —
        it shares silhouette and texture with Pullover, Coat and T-shirt/top, which is where nearly
        all remaining error sits.
      </Caption>
    </Card>
  );
}

function TrainingCurve({ m }: { m: Metrics }) {
  const chart = useChartTheme();
  const h = m.history;
  if (!h?.accuracy?.length) return null;

  const rows = h.accuracy.map((acc, i) => ({
    epoch: i + 1,
    train: acc,
    val: h.val_accuracy?.[i],
  }));
  const phase1 = m.training_config?.phase1_epochs_ran;

  return (
    <Card title="Training history — accuracy per epoch" size="small">
      <div style={{ width: "100%", height: 320 }}>
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={chart.grid} />
            <XAxis dataKey="epoch" stroke={chart.axis} tick={{ fontSize: 11 }} />
            <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} domain={[0.75, 1]} />
            <RTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => pct(n))} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="train" name="Train" stroke={chart.accent} strokeWidth={2} dot={false} />
            <Line
              type="monotone"
              dataKey="val"
              name="Validation"
              stroke={chart.accent2}
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Caption>
        {phase1
          ? `Epochs 1–${phase1} train the classifier head with the MobileNetV2 base frozen; from epoch ${phase1 + 1} the top layers unfreeze at a 100x lower learning rate, which is the step change you can see in the curve.`
          : "Two-phase transfer learning: frozen base, then fine-tuning at a lower learning rate."}
      </Caption>
    </Card>
  );
}

export default function Overview() {
  const navigate = useNavigate();
  const { health } = useBackend();
  const metrics = useAsync(() => api.metrics(), []);
  const conf = useConfColors();

  return (
    <>
      <PageHeader
        eyebrow="MLOps pipeline"
        title="Fashion MNIST classifier"
        subtitle="A MobileNetV2 transfer-learning model serving 10 garment classes over FastAPI, with database-backed retraining. This overview shows the model that is live right now."
        extra={
          <Space wrap>
            <Button type="primary" icon={<ExperimentOutlined />} onClick={() => navigate("/")}>
              Classify an image
            </Button>
            <Button icon={<CloudUploadOutlined />} onClick={() => navigate("/training")}>
              Retrain
            </Button>
          </Space>
        }
      />

      <AsyncSection state={metrics}>
        {(m) => {
          const { base, retrain, current } = headline(m);
          const delta = retrain && base ? (retrain.accuracy - base.accuracy) * 100 : null;
          return (
            <>
              <StatRow>
                <StatCard
                  label="Live accuracy"
                  value={current ? current.accuracy * 100 : 0}
                  precision={2}
                  suffix="%"
                  icon={<AimOutlined />}
                  color={conf.high}
                  hint={
                    retrain
                      ? "Accuracy of the fine-tuned model on its held-out split."
                      : "Accuracy on the 10,000-image held-out test set."
                  }
                />
                <StatCard
                  label="Macro F1"
                  value={current ? current.f1_score : 0}
                  precision={4}
                  icon={<TrophyOutlined />}
                  hint="Unweighted mean of the 10 per-class F1 scores — every class counts equally."
                />
                <StatCard
                  label="Samples awaiting retrain"
                  value={health?.db_samples ?? 0}
                  icon={<DatabaseOutlined />}
                  hint="Labelled rows uploaded to SQLite and not yet used for fine-tuning."
                />
                <StatCard
                  label="API uptime"
                  value={formatUptime(health?.uptime_sec ?? 0)}
                  icon={<ClockCircleOutlined />}
                  hint="Since the container last started. A free Space sleeps after 48 hours idle, which resets this."
                />
              </StatRow>

              {retrain && delta !== null && (
                <Alert
                  style={{ marginBottom: 20 }}
                  type={delta >= 0 ? "success" : "warning"}
                  showIcon
                  message={
                    <span>
                      Model was fine-tuned on {retrain.samples} uploaded sample
                      {retrain.samples === 1 ? "" : "s"} over {retrain.epochs_ran} epoch
                      {retrain.epochs_ran === 1 ? "" : "s"}{" "}
                      <Tag color={delta >= 0 ? "green" : "orange"} style={{ marginInlineStart: 6 }}>
                        {delta >= 0 ? "+" : ""}
                        {delta.toFixed(2)}% vs baseline
                      </Tag>
                    </span>
                  }
                  description={
                    delta < 0
                      ? "Fine-tuning on a small uploaded batch can pull the model away from the full 60,000-image distribution. Compare the per-class figures on Model Metrics before treating this as an improvement."
                      : "Measured on the fine-tuning split, which is much smaller than the original test set — treat it as a signal, not a replacement for the baseline."
                  }
                  action={
                    <Button size="small" onClick={() => navigate("/evaluation")}>
                      Compare
                    </Button>
                  }
                />
              )}

              <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
                <Col xs={24} xl={14}>
                  <PerClassF1 m={m} />
                </Col>
                <Col xs={24} xl={10}>
                  <Card title="Baseline test-set results" size="small" style={{ height: "100%" }}>
                    {base && (
                      <Space direction="vertical" size={14} style={{ width: "100%" }}>
                        {(
                          [
                            ["Accuracy", base.accuracy],
                            ["Macro F1", base.f1_score],
                            ["Precision", base.precision],
                            ["Recall", base.recall],
                          ] as [string, number][]
                        ).map(([label, value]) => (
                          <div key={label}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
                              <Text style={{ fontSize: 13 }}>{label}</Text>
                              <Text strong className="tabular" style={{ fontSize: 13 }}>
                                {value.toFixed(4)}
                              </Text>
                            </div>
                            <Progress
                              percent={value * 100}
                              showInfo={false}
                              size="small"
                              strokeColor={conf.high}
                            />
                          </div>
                        ))}
                        <Paragraph type="secondary" style={{ fontSize: 12, margin: 0, lineHeight: 1.7 }}>
                          Trained on 60,000 images, evaluated on the 10,000-image test split the model
                          never saw. Test loss {base.test_loss?.toFixed(4) ?? "—"}.
                        </Paragraph>
                        <Button block icon={<LineChartOutlined />} onClick={() => navigate("/evaluation")}>
                          Full evaluation
                        </Button>
                      </Space>
                    )}
                  </Card>
                </Col>
              </Row>

              <Row gutter={[16, 16]}>
                <Col xs={24} xl={14}>
                  <TrainingCurve m={m} />
                </Col>
                <Col xs={24} xl={10}>
                  <Card title="The pipeline, end to end" size="small" style={{ height: "100%" }}>
                    <Space direction="vertical" size={12} style={{ width: "100%" }}>
                      {(
                        [
                          ["Classify", "Upload a photo, draw a garment, or pull a held-out test image and see the model's full probability distribution.", "/"],
                          ["Dataset Insights", "Class balance, pixel-intensity distributions and per-class samples from the 70,000-image dataset.", "/dataset"],
                          ["Model Metrics", "Per-class precision/recall/F1, the confusion matrix, and the 30-epoch training history.", "/evaluation"],
                        ] as [string, string, string][]
                      ).map(([title, desc, to]) => (
                        <Card
                          key={to}
                          size="small"
                          hoverable
                          onClick={() => navigate(to)}
                          styles={{ body: { padding: "12px 14px" } }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                            <div style={{ minWidth: 0 }}>
                              <Text strong style={{ fontSize: 13 }}>
                                {title}
                              </Text>
                              <Text type="secondary" style={{ fontSize: 12, display: "block", marginTop: 2, lineHeight: 1.6 }}>
                                {desc}
                              </Text>
                            </div>
                            <ArrowRightOutlined style={{ opacity: 0.4, flexShrink: 0, marginTop: 3 }} />
                          </div>
                        </Card>
                      ))}
                    </Space>
                  </Card>
                </Col>
              </Row>
            </>
          );
        }}
      </AsyncSection>
    </>
  );
}
