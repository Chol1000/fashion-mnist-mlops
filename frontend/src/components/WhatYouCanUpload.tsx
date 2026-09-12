import { Alert, Card, Col, Row, Space, Tag, Typography } from "antd";
import { BulbOutlined } from "@ant-design/icons";
import { CLASS_NAMES } from "../types";

const { Text, Paragraph } = Typography;

/**
 * Sets expectations before someone sends anything to the model.
 *
 * This is a closed-set classifier: softmax over exactly ten garment classes,
 * with no "none of these" output. A photo of a car does not produce a low
 * confidence score or an error — it produces "Bag, 87%", because the model has
 * no way to express that the question was wrong. Saying so up front is the
 * difference between a demo that looks broken and one that looks honest, and
 * it is the single most common way people misread this kind of model.
 *
 * `variant` picks the extra guidance: "image" adds the shot-composition
 * advice that only applies to photographs, "csv" the training-data format
 * rules. "plain" is the closed-set warning alone, for the draw and paste
 * inputs where composition advice makes no sense.
 */
export default function WhatYouCanUpload({
  variant,
  bordered = true,
}: {
  variant: "image" | "csv" | "plain";
  /** False when the caller already provides a card shell around this. */
  bordered?: boolean;
}) {
  const body = (
    <Space direction="vertical" size={14} style={{ width: "100%" }}>
      <div>
        <Text type="secondary" style={{ fontSize: 12.5, lineHeight: 1.7, display: "block", marginBottom: 8 }}>
          Ten garment categories, and nothing else:
        </Text>
        {/* A two-column grid rather than wrapped tags: ten labels of uneven
            width wrap into a ragged block that reads as clutter, and this
            panel now sits in the wide column where a grid has room. */}
        <Row gutter={[8, 8]}>
          {CLASS_NAMES.map((name, i) => (
            <Col key={name} xs={12} sm={8} lg={12} xl={8}>
              <Tag style={{ marginInlineEnd: 0, width: "100%", textAlign: "left" }}>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {i}
                </Text>{" "}
                {name}
              </Tag>
            </Col>
          ))}
        </Row>
      </div>

      {variant !== "csv" && (
        <Alert
          type="warning"
          showIcon
          message="Anything else still gets one of these ten labels"
          description={
            <span>
              There is no “unknown” class. Give it a car, a face or a landscape and the model will not
              refuse — it picks whichever of the ten garments the shape most resembles, sometimes at
              90%+ confidence. A confident answer is not evidence that the input was a garment.
            </span>
          }
        />
      )}

      {variant === "image" && (
        <Paragraph type="secondary" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.75 }}>
          <BulbOutlined style={{ marginInlineEnd: 6 }} />
          <Text strong style={{ fontSize: 12.5 }}>
            For the best result:
          </Text>{" "}
          one garment, centred, filling most of the frame, shot flat against a plain dark background —
          that is what Fashion MNIST looks like. Photos with clutter, strong shadows, a person wearing
          the item, or a light background are out of distribution and classify poorly even when the
          garment is one the model knows.
        </Paragraph>
      )}

      {variant === "csv" && (
        <>
          <Alert
            type="info"
            showIcon
            message="This upload is training data, not images"
            description={
              <span>
                Rows must already be Fashion MNIST-shaped: a <code>label</code> column holding an
                integer 0–9 from the list above, plus <code>pixel1</code> … <code>pixel784</code>{" "}
                holding 0–255 greyscale values for the flattened 28x28 image. Photos, ZIPs and folders
                of images are not accepted — and rows whose label falls outside 0–9 are dropped by the
                API rather than silently trained on.
              </span>
            }
          />
          <Paragraph type="secondary" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.75 }}>
            Mislabelled rows are the one thing nothing here can catch: the API validates the shape and
            the label range, but it cannot tell that you labelled a Sandal as a Bag. Fine-tuning on
            wrong labels will make the model measurably worse, so use the sample CSV on the right if
            you only want to watch the pipeline run.
          </Paragraph>
        </>
      )}
    </Space>
  );

  if (!bordered) return body;

  return (
    <Card size="small" type="inner" title="What this model can classify">
      {body}
    </Card>
  );
}
