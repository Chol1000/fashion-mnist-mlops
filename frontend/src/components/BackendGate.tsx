import { useState, type ReactNode } from "react";
import { Button, Card, Progress, Space, Spin, Typography, Alert } from "antd";
import { CloudServerOutlined, ReloadOutlined, ApiOutlined } from "@ant-design/icons";
import { useBackend } from "../context/BackendContext";
import { API_IS_REMOTE } from "../api";
import { COLD_START_SECONDS, SPACE_URL } from "../config";

const { Title, Text, Paragraph } = Typography;

/**
 * Holds the app at a splash screen until the API answers for the first time.
 *
 * The reason this exists rather than letting each page show its own error: on
 * a free Hugging Face Space the container sleeps after 48h idle, so the first
 * visitor after a quiet weekend lands while TensorFlow is still importing.
 * Without this the whole dashboard renders with an error in every card, which
 * reads as "the project is broken" rather than "give it a minute". The polling
 * in BackendContext also keeps the request pressure on, which is itself what
 * wakes a sleeping Space.
 *
 * It only blocks the *first* connection. Once the API has answered this
 * session, a later blip shows an inline banner instead — by then the user has
 * work on screen worth keeping.
 */
export default function BackendGate({ children }: { children: ReactNode }) {
  const { online, everOnline, downForSec, attempts, refresh } = useBackend();
  const [skipped, setSkipped] = useState(false);

  if (online === true || everOnline || skipped) {
    return (
      <>
        {online === false && (
          <Alert
            type="warning"
            showIcon
            banner
            message="Lost contact with the API — retrying automatically."
            style={{ position: "sticky", top: 0, zIndex: 100 }}
          />
        )}
        {children}
      </>
    );
  }

  // First probe still in flight: a bare spinner, because at this point there
  // is no evidence anything is wrong and an alarming message would be wrong
  // most of the time.
  if (online === null) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <Spin size="large" tip="Connecting to the API…">
          <div style={{ padding: 40 }} />
        </Spin>
      </div>
    );
  }

  // Capped so the bar keeps creeping without ever claiming to be finished —
  // a cold start has no progress signal to report, only elapsed time.
  const pct = Math.min(95, Math.round((downForSec / COLD_START_SECONDS) * 100));

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        padding: 16,
      }}
    >
      <Card style={{ maxWidth: 560, width: "100%" }} styles={{ body: { padding: "32px 28px" } }}>
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Space align="start" size={14}>
            <CloudServerOutlined style={{ fontSize: 30, color: "var(--color-primary)" }} />
            <div>
              <Title level={4} style={{ margin: 0 }}>
                Waking the prediction API
              </Title>
              <Text type="secondary" style={{ fontSize: 13 }}>
                {downForSec}s elapsed · {attempts} {attempts === 1 ? "attempt" : "attempts"}
              </Text>
            </div>
          </Space>

          <Progress percent={pct} status="active" showInfo={false} />

          <Paragraph type="secondary" style={{ fontSize: 13, marginBottom: 0, lineHeight: 1.7 }}>
            The backend runs on a free Hugging Face Space, which sleeps after 48 hours without
            traffic. This page is already knocking on it — a sleeping Space wakes on the first
            request it receives. Starting the container, importing TensorFlow and loading the
            MobileNetV2 weights usually takes about {COLD_START_SECONDS} seconds; nothing else is
            needed from you.
          </Paragraph>

          {downForSec > COLD_START_SECONDS * 2 && (
            <Alert
              type="info"
              showIcon
              message="Taking longer than a normal cold start"
              description={
                API_IS_REMOTE
                  ? "A sleeping Space wakes on its own from the requests this page is already sending. One that is paused or has crashed cannot — that needs an owner to restart it. Use the button below, then Settings → Restart this Space."
                  : "The container may have failed to start rather than simply being asleep. Open the Space below and check its build and container logs."
              }
            />
          )}

          <Space wrap>
            <Button type="primary" icon={<ReloadOutlined />} onClick={refresh}>
              Retry now
            </Button>
            <Button icon={<ApiOutlined />} href={SPACE_URL} target="_blank" rel="noreferrer">
              Restart it on Hugging Face
            </Button>
            <Button type="text" onClick={() => setSkipped(true)}>
              Continue anyway
            </Button>
          </Space>
        </Space>
      </Card>
    </div>
  );
}
