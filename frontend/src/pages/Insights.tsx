import { Card, Col, Row, Table, Tag, Typography, Space } from "antd";
import { PictureOutlined, AppstoreOutlined, ExpandOutlined, BgColorsOutlined } from "@ant-design/icons";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip, ReferenceLine,
} from "recharts";
import { api } from "../api";
import { useAsync } from "../hooks";
import {
  PageHeader, StatCard, StatRow, Caption, FigureCard, useChartTheme, tooltipValue, AsyncSection,
} from "../ui";

const { Paragraph, Text } = Typography;

export default function Insights() {
  const insights = useAsync(() => api.insights(), []);
  const chart = useChartTheme();

  return (
    <>
      <PageHeader
        eyebrow="Exploratory analysis"
        title="Dataset insights"
        subtitle="Fashion MNIST: 70,000 greyscale images across 10 garment categories, 60,000 for training and 10,000 held out. The three features below are what actually explain the model's error profile."
      />

      <AsyncSection state={insights}>
        {(d) => {
          const rows = Object.entries(d.classes).map(([id, c]) => ({
            key: id,
            id: Number(id),
            name: c.name,
            count: c.count,
            share: c.count / d.total_train_samples,
          }));
          const mean = d.total_train_samples / d.num_classes;

          return (
            <>
              <StatRow>
                <StatCard
                  label="Training images"
                  value={d.total_train_samples.toLocaleString()}
                  icon={<PictureOutlined />}
                />
                <StatCard label="Classes" value={d.num_classes} icon={<AppstoreOutlined />} />
                <StatCard label="Image size" value={d.image_size} icon={<ExpandOutlined />} />
                <StatCard
                  label="Mean pixel value"
                  value={d.pixel_statistics.mean}
                  precision={1}
                  icon={<BgColorsOutlined />}
                  hint={`Std ${d.pixel_statistics.std.toFixed(1)}, range ${d.pixel_statistics.min}–${d.pixel_statistics.max}. Low mean because most of every image is black background.`}
                />
              </StatRow>

              <Card title="Feature 1 — class distribution" size="small" style={{ marginBottom: 20 }}>
                <Paragraph type="secondary" style={{ fontSize: 13, marginTop: 0, lineHeight: 1.7 }}>
                  The dataset is exactly balanced: {mean.toLocaleString()} training images per class.
                  That matters for how the results on Model Metrics should be read — with no class
                  imbalance to correct for, macro F1 and plain accuracy tell the same story, and any
                  per-class weakness is about visual similarity rather than missing data.
                </Paragraph>

                <Row gutter={[16, 16]}>
                  <Col xs={24} xl={15}>
                    <div style={{ width: "100%", height: 320 }}>
                      <ResponsiveContainer>
                        <BarChart data={rows} margin={{ top: 8, right: 12, left: 4, bottom: 56 }}>
                          <CartesianGrid stroke={chart.grid} vertical={false} />
                          <XAxis
                            dataKey="name"
                            stroke={chart.axis}
                            tick={{ fontSize: 11 }}
                            angle={-35}
                            textAnchor="end"
                            interval={0}
                          />
                          <YAxis stroke={chart.axis} tick={{ fontSize: 11 }} />
                          <RTooltip
                            contentStyle={chart.tooltip}
                            cursor={{ fill: chart.dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }}
                            formatter={tooltipValue((n) => n.toLocaleString(), "Images")}
                          />
                          <ReferenceLine
                            y={mean}
                            stroke={chart.accent2}
                            strokeDasharray="4 4"
                            label={{ value: "mean", fontSize: 11, fill: chart.axis, position: "right" }}
                          />
                          <Bar dataKey="count" fill={chart.accent} radius={[4, 4, 0, 0]} maxBarSize={48} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </Col>
                  <Col xs={24} xl={9}>
                    <Table
                      size="small"
                      dataSource={rows}
                      pagination={false}
                      scroll={{ y: 300 }}
                      columns={[
                        {
                          title: "Class",
                          dataIndex: "name",
                          key: "name",
                          render: (v: string, r) => (
                            <Space size={6}>
                              <Tag style={{ marginInlineEnd: 0 }}>{r.id}</Tag>
                              <Text style={{ fontSize: 13 }}>{v}</Text>
                            </Space>
                          ),
                        },
                        {
                          title: "Images",
                          dataIndex: "count",
                          key: "count",
                          align: "right",
                          render: (v: number) => <span className="tabular">{v.toLocaleString()}</span>,
                        },
                        {
                          title: "Share",
                          dataIndex: "share",
                          key: "share",
                          align: "right",
                          render: (v: number) => <span className="tabular">{(v * 100).toFixed(1)}%</span>,
                        },
                      ]}
                    />
                  </Col>
                </Row>
                <Caption>
                  Label ids are the integers the <code>label</code> column of an upload CSV must use.
                </Caption>
              </Card>

              <FigureCard
                title="Feature 2 — pixel intensity distribution per class"
                src={api.figure("eda_03_pixel_intensity.png")}
                caption="Brightness distribution per category. Trousers and Bags skew dark and separate cleanly; Shirt, Pullover and Coat overlap heavily, which is precisely the group the confusion matrix shows the model struggling with."
              />

              <FigureCard
                title="Feature 3 — sample images per class"
                src={api.figure("eda_02_sample_images.png")}
                caption="Visual inspection explains per-class difficulty better than any statistic: Pullover, Coat and Shirt share collar and sleeve shapes at 28x28, while Bag and Trouser have silhouettes nothing else resembles."
              />

              <FigureCard
                title="Preprocessing pipeline"
                src={api.figure("preprocessing_pipeline.png")}
                caption="What happens to one image between the CSV row and the model's input tensor — the same transforms the Upload & Retrain page applies to anything you upload."
              />
            </>
          );
        }}
      </AsyncSection>
    </>
  );
}
