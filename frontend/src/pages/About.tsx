import { Card, Col, Row, Typography, Space, Tag, Button, Descriptions, Steps } from "antd";
import { GithubOutlined, ApiOutlined, CloudServerOutlined } from "@ant-design/icons";
import { api } from "../api";
import { REPO_URL, SPACE_URL } from "../config";
import { PageHeader } from "../ui";
import { CLASS_NAMES } from "../types";

const { Paragraph, Text, Title } = Typography;

const STACK: [string, string][] = [
  ["Model", "TensorFlow / Keras · MobileNetV2 transfer learning, two-phase schedule"],
  ["Serving", "FastAPI + Uvicorn, single worker, model cached in process"],
  ["Dashboard", "React 19, Vite, Ant Design, Recharts"],
  ["Storage", "SQLite for uploaded retraining samples and run history"],
  ["Packaging", "Docker, one image serving both the API and this dashboard"],
  ["Load testing", "Locust, against both the local stack and the deployed Space"],
];

export default function About() {
  return (
    <>
      <PageHeader
        eyebrow="About"
        title="How this project fits together"
        subtitle="An end-to-end machine learning pipeline: data preparation, transfer learning, a serving API, this dashboard, database-backed retraining, containerisation, and a deployment that can be load tested."
        extra={
          <Space wrap>
            <Button icon={<GithubOutlined />} href={REPO_URL} target="_blank" rel="noreferrer">
              Source
            </Button>
            <Button icon={<ApiOutlined />} href={api.docsUrl()} target="_blank" rel="noreferrer">
              API docs
            </Button>
          </Space>
        }
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={14}>
          <Card size="small" title="The loop" style={{ marginBottom: 16 }}>
            <Steps
              direction="vertical"
              size="small"
              current={-1}
              items={[
                {
                  title: "Train",
                  description:
                    "MobileNetV2 pre-trained on ImageNet, with a new classifier head. Phase one trains the head with the base frozen; phase two unfreezes the top layers at a 100x lower learning rate. 60,000 images in, 93.17% accuracy on 10,000 held out.",
                },
                {
                  title: "Serve",
                  description:
                    "FastAPI loads the saved checkpoint once at startup and holds it in memory. Predictions accept either 784 raw pixel values or an uploaded image, which the API converts to the model's input shape itself.",
                },
                {
                  title: "Collect",
                  description:
                    "Labelled CSV rows uploaded through the API are validated, cleaned and written to SQLite — so retraining data accumulates across requests and survives a restart of the training process.",
                },
                {
                  title: "Retrain",
                  description:
                    "Fine-tuning runs as a background task against whatever is in the database, streaming per-epoch metrics back to the dashboard. When it finishes, the serving process reloads the new weights without a restart.",
                },
                {
                  title: "Observe",
                  description:
                    "Evaluation metrics, per-class breakdowns and run history are all exposed as endpoints, which is what every chart on this dashboard reads from.",
                },
              ]}
            />
          </Card>

          <Card size="small" title="The ten classes">
            <Space size={[6, 6]} wrap>
              {CLASS_NAMES.map((name, i) => (
                <Tag key={name} style={{ marginInlineEnd: 0 }}>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    {i}
                  </Text>{" "}
                  {name}
                </Tag>
              ))}
            </Space>
            <Paragraph type="secondary" style={{ fontSize: 12.5, marginTop: 12, marginBottom: 0, lineHeight: 1.75 }}>
              Fashion MNIST was published by Zalando Research as a drop-in replacement for the original
              handwritten-digit MNIST — same 28x28 greyscale format and same 70,000 images, but a
              harder problem, because several of the ten categories genuinely overlap at this
              resolution.
            </Paragraph>
          </Card>
        </Col>

        <Col xs={24} xl={10}>
          <Card size="small" title="Stack" style={{ marginBottom: 16 }}>
            <Descriptions
              size="small"
              column={1}
              items={STACK.map(([label, children]) => ({ key: label, label, children }))}
            />
          </Card>

          <Card size="small" title="Deployment">
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <Paragraph type="secondary" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.75 }}>
                One Docker image contains the API and this dashboard: Vite builds the frontend, FastAPI
                serves the result alongside its own routes. That is deliberate — it means there is a
                single service to deploy, a single URL, and no second service that can be asleep when
                the first one is awake.
              </Paragraph>
              <Paragraph type="secondary" style={{ fontSize: 12.5, margin: 0, lineHeight: 1.75 }}>
                The same image runs unchanged under Docker Compose behind an Nginx load balancer, which
                is how the Locust results in the repository were produced.
              </Paragraph>
              <Button icon={<CloudServerOutlined />} href={SPACE_URL} target="_blank" rel="noreferrer" block>
                Open the Space
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      <Title level={5} style={{ marginTop: 24 }}>
        Credits
      </Title>
      <Paragraph type="secondary" style={{ fontSize: 12.5, lineHeight: 1.8, maxWidth: 820 }}>
        Dataset: Fashion MNIST, Zalando Research (MIT licence). Base model: MobileNetV2, pre-trained on
        ImageNet. Everything else — preprocessing, training schedule, API, retraining loop and this
        dashboard — is in the repository linked above.
      </Paragraph>
    </>
  );
}
