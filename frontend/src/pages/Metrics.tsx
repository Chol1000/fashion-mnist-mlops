import { Card, Col, Row, Table, Tag, Typography, Space, Descriptions, Alert, Progress } from "antd";
import { AimOutlined, TrophyOutlined, FunctionOutlined, RadarChartOutlined } from "@ant-design/icons";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  LineChart, Line, Legend, Cell, ReferenceLine,
} from "recharts";
import { api } from "../api";
import { useAsync, pct } from "../hooks";
import {
  PageHeader, StatCard, StatRow, Caption, FigureCard, useChartTheme, useConfColors,
  tooltipValue, AsyncSection,
} from "../ui";
import type { Metrics as MetricsT } from "../types";

const { Paragraph, Text } = Typography;

function ConfigCard({ m }: { m: MetricsT }) {
  const cfg = m.training_config;
  if (!cfg) return null;
  const totalEpochs = (cfg.phase1_epochs_ran ?? 0) + (cfg.phase2_epochs_ran ?? 0);

  return (
    <Card title="Architecture & training configuration" size="small" style={{ marginBottom: 20 }}>
      <Descriptions
        size="small"
        column={{ xs: 1, sm: 2, xl: 3 }}
        items={[
          { key: "model", label: "Base model", children: cfg.model ?? "MobileNetV2" },
          {
            key: "input",
            label: "Input shape",
            children: (cfg.input_shape ?? [128, 128, 3]).join(" x "),
          },
          { key: "epochs", label: "Epochs ran", children: totalEpochs || "—" },
          { key: "batch", label: "Batch size", children: cfg.batch_size ?? "—" },
          {
            key: "val",
            label: "Validation split",
            children: cfg.val_split != null ? `${(cfg.val_split * 100).toFixed(0)}%` : "—",
          },
          {
            key: "phases",
            label: "Two-phase schedule",
            children: `${cfg.phase1_epochs_ran ?? 0} frozen + ${cfg.phase2_epochs_ran ?? 0} fine-tune`,
          },
          {
            key: "head",
            label: "Classifier head",
            span: { xs: 1, sm: 2, xl: 3 },
            children: <code style={{ fontSize: 12 }}>{cfg.head ?? "—"}</code>,
          },
          {
            key: "opt",
            label: "Optimisers",
            span: { xs: 1, sm: 2, xl: 3 },
            children: (
              <Space size={[6, 6]} wrap>
                {cfg.optimizer_phase1 && <Tag>Phase 1 · {cfg.optimizer_phase1}</Tag>}
                {cfg.optimizer_phase2 && <Tag>Phase 2 · {cfg.optimizer_phase2}</Tag>}
              </Space>
            ),
          },
          {
            key: "cb",
            label: "Callbacks",
            span: { xs: 1, sm: 2, xl: 3 },
            children: (
              <Space size={[6, 6]} wrap>
                {(cfg.callbacks ?? []).map((c) => (
                  <Tag key={c}>{c}</Tag>
                ))}
              </Space>
            ),
          },
          {
            key: "aug",
            label: "Augmentation",
            span: { xs: 1, sm: 2, xl: 3 },
            children: (
              <Space size={[6, 6]} wrap>
                {(cfg.augmentation ?? []).map((a) => (
                  <Tag key={a}>{a}</Tag>
                ))}
              </Space>
            ),
          },
          {
            key: "reg",
            label: "Regularisation",
            span: { xs: 1, sm: 2, xl: 3 },
            children: "L2 weight decay on the Dense(256) head · Dropout(0.5) · BatchNormalization",
          },
        ]}
      />
    </Card>
  );
}

function PerClassTable({ m }: { m: MetricsT }) {
  const chart = useChartTheme();
  const conf = useConfColors();
  const ev = m.evaluation ?? m.initial_training;
  const f1 = ev?.per_class_f1;
  if (!f1) return null;

  const rows = Object.keys(f1).map((name) => ({
    key: name,
    name,
    f1: f1[name],
    precision: ev?.per_class_precision?.[name],
    recall: ev?.per_class_recall?.[name],
  }));
  const sorted = [...rows].sort((a, b) => b.f1 - a.f1);
  const mean = rows.reduce((s, r) => s + r.f1, 0) / rows.length;

  return (
    <Card title="Per-class evaluation — all 10 categories" size="small" style={{ marginBottom: 20 }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={13}>
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={sorted} margin={{ top: 8, right: 12, left: 0, bottom: 56 }}>
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
                <ReferenceLine
                  y={mean}
                  stroke={chart.accent2}
                  strokeDasharray="4 4"
                  label={{ value: `macro ${mean.toFixed(3)}`, fontSize: 11, fill: chart.axis, position: "right" }}
                />
                <Bar dataKey="f1" radius={[4, 4, 0, 0]} maxBarSize={46}>
                  {sorted.map((r) => (
                    <Cell key={r.name} fill={r.f1 >= 0.95 ? conf.high : r.f1 >= 0.85 ? chart.accent : conf.mid} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Col>
        <Col xs={24} xl={11}>
          <Table
            size="small"
            dataSource={sorted}
            pagination={false}
            scroll={{ y: 320 }}
            columns={[
              { title: "Class", dataIndex: "name", key: "name" },
              {
                title: "F1",
                dataIndex: "f1",
                key: "f1",
                align: "right",
                sorter: (a, b) => a.f1 - b.f1,
                render: (v: number) => <span className="tabular">{v.toFixed(4)}</span>,
              },
              {
                title: "Precision",
                dataIndex: "precision",
                key: "precision",
                align: "right",
                render: (v?: number) => <span className="tabular">{v?.toFixed(4) ?? "—"}</span>,
              },
              {
                title: "Recall",
                dataIndex: "recall",
                key: "recall",
                align: "right",
                render: (v?: number) => <span className="tabular">{v?.toFixed(4) ?? "—"}</span>,
              },
            ]}
          />
        </Col>
      </Row>
      <Caption>
        Precision is "when it says Shirt, how often is it right"; recall is "of the real Shirts, how
        many did it find". Shirt is weak on both — it is not being over- or under-predicted so much as
        genuinely confused with the three other upper-body garments.
      </Caption>
    </Card>
  );
}

function History({ m }: { m: MetricsT }) {
  const chart = useChartTheme();
  const h = m.history;
  if (!h?.accuracy?.length) return null;
  const phase1 = m.training_config?.phase1_epochs_ran;

  const accRows = h.accuracy.map((v, i) => ({ epoch: i + 1, train: v, val: h.val_accuracy?.[i] }));
  const lossRows = (h.loss ?? []).map((v, i) => ({ epoch: i + 1, train: v, val: h.val_loss?.[i] }));

  const marker = phase1 ? (
    <ReferenceLine
      x={phase1}
      stroke={chart.axis}
      strokeDasharray="3 3"
      label={{ value: "unfreeze", fontSize: 10, fill: chart.axis, position: "top" }}
    />
  ) : null;

  return (
    <Card title={`${h.accuracy.length}-epoch training history`} size="small" style={{ marginBottom: 20 }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Text strong style={{ fontSize: 13 }}>
            Accuracy
          </Text>
          <div style={{ width: "100%", height: 280, marginTop: 8 }}>
            <ResponsiveContainer>
              <LineChart data={accRows} margin={{ top: 12, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid stroke={chart.grid} />
                <XAxis dataKey="epoch" stroke={chart.axis} tick={{ fontSize: 11 }} />
                <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} domain={[0.75, 1]} />
                <RTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => pct(n))} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {marker}
                <Line dataKey="train" name="Train" stroke={chart.accent} strokeWidth={2} dot={false} />
                <Line
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
        </Col>
        <Col xs={24} xl={12}>
          <Text strong style={{ fontSize: 13 }}>
            Loss
          </Text>
          <div style={{ width: "100%", height: 280, marginTop: 8 }}>
            <ResponsiveContainer>
              <LineChart data={lossRows} margin={{ top: 12, right: 12, left: -10, bottom: 0 }}>
                <CartesianGrid stroke={chart.grid} />
                <XAxis dataKey="epoch" stroke={chart.axis} tick={{ fontSize: 11 }} />
                <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} />
                <RTooltip contentStyle={chart.tooltip} formatter={tooltipValue((n) => n.toFixed(4))} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {marker}
                <Line dataKey="train" name="Train" stroke={chart.accent} strokeWidth={2} dot={false} />
                <Line
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
        </Col>
      </Row>
      <Caption>
        {phase1
          ? `Validation accuracy sits above training accuracy throughout, which is the expected signature of dropout and augmentation being active during training but not evaluation — not a data leak. The step at epoch ${phase1} is where the MobileNetV2 base unfreezes.`
          : "Validation above training is the expected signature of dropout and augmentation being active during training but not evaluation."}
      </Caption>
    </Card>
  );
}

export default function Metrics() {
  const metrics = useAsync(() => api.metrics(), []);
  const conf = useConfColors();

  return (
    <>
      <PageHeader
        eyebrow="Evaluation"
        title="Model metrics"
        subtitle="Everything the training run recorded: headline scores on the held-out test split, a per-class breakdown, the confusion matrix, and the full epoch history."
      />

      <AsyncSection state={metrics}>
        {(m) => {
          const base = m.evaluation ?? m.initial_training;
          const rt = m.retrain;
          const delta = rt && base ? (rt.accuracy - base.accuracy) * 100 : null;

          return (
            <>
              <ConfigCard m={m} />

              {base && (
                <>
                  <Text
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      color: "var(--color-primary)",
                      display: "block",
                      marginBottom: 10,
                    }}
                  >
                    Baseline — 60,000 train / 10,000 held-out test
                  </Text>
                  <StatRow cols={5}>
                    <StatCard
                      label="Test accuracy"
                      value={base.accuracy * 100}
                      precision={2}
                      suffix="%"
                      icon={<AimOutlined />}
                      color={conf.high}
                    />
                    <StatCard
                      label="Test loss"
                      value={base.test_loss ?? 0}
                      precision={4}
                      icon={<FunctionOutlined />}
                    />
                    <StatCard label="Macro F1" value={base.f1_score} precision={4} icon={<TrophyOutlined />} />
                    <StatCard label="Precision" value={base.precision} precision={4} icon={<RadarChartOutlined />} />
                    <StatCard label="Recall" value={base.recall} precision={4} icon={<RadarChartOutlined />} />
                  </StatRow>
                </>
              )}

              {rt && delta !== null && (
                <Card
                  size="small"
                  style={{ marginBottom: 20 }}
                  title={
                    <Space wrap>
                      <span>After fine-tuning</span>
                      <Tag color="blue">{rt.samples} uploaded samples</Tag>
                      <Tag color="blue">{rt.epochs_ran} epochs</Tag>
                      <Tag color={delta >= 0 ? "green" : "orange"}>
                        {delta >= 0 ? "+" : ""}
                        {delta.toFixed(2)}% vs baseline
                      </Tag>
                    </Space>
                  }
                >
                  <Alert
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                    message="Not a like-for-like comparison"
                    description="These numbers come from the validation split of your uploaded batch, which is far smaller than the 10,000-image test set above and drawn from a different sample. A higher number here does not by itself mean the model got better."
                  />
                  <Row gutter={[16, 16]}>
                    {(
                      [
                        ["Accuracy", rt.accuracy, true],
                        ["Macro F1", rt.f1_score, false],
                        ["Precision", rt.precision, false],
                        ["Recall", rt.recall, false],
                      ] as [string, number, boolean][]
                    ).map(([label, value, isPct]) => (
                      <Col key={label} xs={12} xl={6}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {label}
                        </Text>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                          <Text strong className="tabular" style={{ fontSize: 20 }}>
                            {isPct ? pct(value) : value.toFixed(4)}
                          </Text>
                        </div>
                        <Progress percent={value * 100} showInfo={false} size="small" strokeColor={conf.high} />
                      </Col>
                    ))}
                  </Row>
                </Card>
              )}

              <PerClassTable m={m} />

              <FigureCard
                title="Confusion matrix — test set"
                src={api.figure("confusion_matrix.png")}
                caption="Off-diagonal mass is concentrated in one block: Shirt against T-shirt/top, Pullover and Coat. Footwear (Sandal, Sneaker, Ankle boot) forms a second, much smaller cluster. Everything else is essentially solved."
              />

              <History m={m} />

              <Row gutter={[16, 16]}>
                <Col xs={24} xl={12}>
                  <FigureCard
                    title="Per-class metrics"
                    src={api.figure("per_class_metrics.png")}
                    minWidth={420}
                    caption="Precision, recall and F1 side by side for each category."
                  />
                </Col>
                <Col xs={24} xl={12}>
                  <FigureCard
                    title="Confidence analysis"
                    src={api.figure("confidence_analysis.png")}
                    minWidth={420}
                    caption="How the model's confidence relates to whether it was right — the basis for the thresholds the Classify page colours predictions by."
                  />
                </Col>
              </Row>

              <Paragraph type="secondary" style={{ fontSize: 12, lineHeight: 1.8 }}>
                Figures are regenerated by the training notebook and served as static files by the API,
                so they always correspond to the checkpoint currently in <code>models/</code>.
              </Paragraph>
            </>
          );
        }}
      </AsyncSection>
    </>
  );
}
