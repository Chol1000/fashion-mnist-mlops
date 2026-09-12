import { Card, Col, Row, Space, Statistic, Tag, Typography } from "antd";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, Cell,
} from "recharts";
import { ConfidenceTag, Caption, useChartTheme, useConfColor, tooltipValue } from "../ui";
import { pct } from "../hooks";
import type { PredictResult } from "../types";

const { Text } = Typography;

/**
 * The model's answer plus the full posterior.
 *
 * Showing all ten probabilities, not just the winner, is the point: a 0.42 top
 * class with 0.39 runner-up is a materially different result from a 0.99, and
 * the runner-up is usually the visually adjacent garment (Shirt vs
 * T-shirt/top) which tells you *why* it hesitated.
 */
export default function PredictionResult({
  result,
  trueLabel,
}: {
  result: PredictResult;
  /** Set only for held-out test samples, where ground truth is known. */
  trueLabel?: string;
}) {
  const chart = useChartTheme();
  const color = useConfColor(result.confidence);

  const rows = Object.entries(result.probabilities)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const runnerUp = rows[1];
  const correct = trueLabel === undefined ? undefined : result.predicted_label === trueLabel;

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card size="small" styles={{ body: { padding: "20px 22px" } }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} md={13}>
            <Space direction="vertical" size={6}>
              <Text
                type="secondary"
                style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}
              >
                Predicted class
              </Text>
              <Text strong style={{ fontSize: 30, lineHeight: 1.15, display: "block" }}>
                {result.predicted_label}
              </Text>
              <Space size={8} wrap>
                <ConfidenceTag confidence={result.confidence} />
                {correct !== undefined && (
                  <Tag color={correct ? "green" : "red"} style={{ marginInlineEnd: 0, fontWeight: 600 }}>
                    {correct ? "Matches true label" : `True label: ${trueLabel}`}
                  </Tag>
                )}
              </Space>
            </Space>
          </Col>
          <Col xs={24} md={11}>
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title={<span style={{ fontSize: 12 }}>Confidence</span>}
                  value={result.confidence * 100}
                  precision={1}
                  suffix="%"
                  styles={{ content: { fontSize: 24, fontWeight: 700, color } }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title={<span style={{ fontSize: 12 }}>Runner-up</span>}
                  value={runnerUp ? runnerUp.value * 100 : 0}
                  precision={1}
                  suffix="%"
                  styles={{ content: { fontSize: 24, fontWeight: 700 } }}
                />
                {runnerUp && (
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {runnerUp.name}
                  </Text>
                )}
              </Col>
            </Row>
          </Col>
        </Row>
      </Card>

      <Card title="Probability across all 10 classes" size="small">
        <div style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 46, left: 8, bottom: 4 }}>
              <CartesianGrid stroke={chart.grid} horizontal={false} />
              <XAxis type="number" domain={[0, 1]} stroke={chart.axis} tick={{ fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="name"
                width={92}
                stroke={chart.axis}
                tick={{ fontSize: 11 }}
                interval={0}
              />
              <RTooltip
                contentStyle={chart.tooltip}
                cursor={{ fill: chart.dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }}
                formatter={tooltipValue((n) => pct(n, 2), "Probability")}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                {rows.map((r, i) => (
                  // Only the winner is coloured; the rest stay muted so the
                  // shape of the distribution reads at a glance.
                  <Cell key={r.name} fill={i === 0 ? color : chart.muted} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Caption>
          Softmax output, so the ten bars sum to 1. A tall second bar means the model found the image
          genuinely ambiguous rather than simply getting it wrong.
        </Caption>
      </Card>
    </Space>
  );
}
