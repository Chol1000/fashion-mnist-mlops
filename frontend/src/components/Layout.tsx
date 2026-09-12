import { useMemo, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Layout as AntLayout, Menu, Button, Typography, Space, Drawer, Grid, Badge, Tooltip,
} from "antd";
import {
  DashboardOutlined, ExperimentOutlined, BarChartOutlined,
  LineChartOutlined, ApiOutlined, InfoCircleOutlined, SunOutlined, MoonOutlined,
  MenuOutlined, FileTextOutlined,
} from "@ant-design/icons";
import { useThemeMode } from "../context/ThemeContext";
import { useBackend } from "../context/BackendContext";
import { api } from "../api";

const { Sider, Header, Content } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  {
    type: "group" as const,
    label: "Serving",
    children: [
      { key: "/", icon: <ExperimentOutlined />, label: "Classify" },
      { key: "/overview", icon: <DashboardOutlined />, label: "Overview" },
    ],
  },
  {
    type: "group" as const,
    label: "Pipeline",
    children: [
      { key: "/dataset", icon: <BarChartOutlined />, label: "Dataset Insights" },
      { key: "/evaluation", icon: <LineChartOutlined />, label: "Model Metrics" },
    ],
  },
  {
    type: "group" as const,
    label: "Operations",
    children: [
      { key: "/status", icon: <ApiOutlined />, label: "API & System" },
      { key: "/about", icon: <InfoCircleOutlined />, label: "About" },
    ],
  },
];

/** Live heartbeat against the API, shown as a coloured dot in the header so
 *  you can tell at a glance whether what's on screen is current or a stale
 *  render of a backend that went to sleep. */
function SystemStatus({ compact }: { compact: boolean }) {
  const { online, health } = useBackend();

  const status = online === null ? "default" : online ? (health?.model_ready ? "success" : "warning") : "error";
  const label =
    online === null
      ? "Connecting…"
      : !online
        ? "API asleep — retrying"
        : health?.model_ready
          ? "API live · model loaded"
          : "API live · model not loaded";

  if (compact) {
    return (
      <Tooltip title={label}>
        <Badge status={status} />
      </Tooltip>
    );
  }
  return (
    <Space size={6}>
      <Badge status={status} />
      <Text type="secondary" style={{ fontSize: 13, whiteSpace: "nowrap" }}>
        {label}
      </Text>
    </Space>
  );
}

function SidebarNav({ activeKey, onNavigate }: { activeKey: string; onNavigate?: () => void }) {
  const navigate = useNavigate();
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "16px 20px",
          borderBottom: "1px solid var(--color-border)",
          flexShrink: 0,
        }}
      >
        <div style={{ minWidth: 0, lineHeight: 1.25 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: "var(--color-primary)" }}>Fashion MNIST</div>
          <Text type="secondary" style={{ fontSize: 11 }}>
            MLOps Pipeline
          </Text>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
        <Menu
          mode="inline"
          selectedKeys={[activeKey]}
          items={NAV_ITEMS}
          onClick={({ key }) => {
            navigate(key);
            onNavigate?.();
          }}
          style={{ borderInlineEnd: 0 }}
        />
      </div>

      <div
        style={{
          flexShrink: 0,
          padding: "12px 20px 16px",
          borderTop: "1px solid var(--color-border)",
          lineHeight: 1.6,
        }}
      >
        <Text type="secondary" style={{ fontSize: 11 }}>
          MobileNetV2 transfer learning · 10 garment classes · 93.17% test accuracy. Retraining runs
          against uploaded samples held in SQLite.
        </Text>
      </div>
    </div>
  );
}

export default function Layout() {
  const location = useLocation();
  const { mode, toggle } = useThemeMode();
  const screens = Grid.useBreakpoint();
  const isMobile = !screens.lg;
  const [menuOpen, setMenuOpen] = useState(false);

  const activeKey = useMemo(() => {
    const p = location.pathname;
    return p === "/" ? "/" : "/" + p.split("/")[1];
  }, [location.pathname]);

  return (
    <AntLayout style={{ minHeight: "100vh" }}>
      {!isMobile && (
        <Sider
          width={248}
          theme="light"
          style={{
            borderInlineEnd: "1px solid var(--color-border)",
            position: "sticky",
            top: 0,
            height: "100vh",
            background: "var(--color-surface)",
          }}
        >
          <SidebarNav activeKey={activeKey} />
        </Sider>
      )}

      {isMobile && (
        <Drawer
          placement="left"
          size={248}
          open={menuOpen}
          onClose={() => setMenuOpen(false)}
          closable={false}
          styles={{ body: { padding: 0, height: "100%" } }}
        >
          <SidebarNav activeKey={activeKey} onNavigate={() => setMenuOpen(false)} />
        </Drawer>
      )}

      <AntLayout>
        <Header
          style={{
            background: "var(--color-surface)",
            borderBottom: "1px solid var(--color-border)",
            display: "flex",
            alignItems: "center",
            gap: isMobile ? 8 : 16,
            padding: isMobile ? "0 12px" : "0 28px",
            height: 60,
            lineHeight: "normal",
            position: "sticky",
            top: 0,
            zIndex: 10,
          }}
        >
          {isMobile && (
            <Button shape="circle" icon={<MenuOutlined />} onClick={() => setMenuOpen(true)} style={{ flexShrink: 0 }} />
          )}

          {!isMobile && (
            <Text strong style={{ fontSize: 15 }}>
              Fashion MNIST Classifier
            </Text>
          )}

          <Space size={isMobile ? 8 : 16} style={{ marginInlineStart: "auto", flexShrink: 0 }}>
            <SystemStatus compact={isMobile} />
            {!isMobile && (
              <Button icon={<FileTextOutlined />} href={api.docsUrl()} target="_blank" rel="noreferrer">
                API Docs
              </Button>
            )}
            <Tooltip title={mode === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
              <Button shape="circle" icon={mode === "dark" ? <SunOutlined /> : <MoonOutlined />} onClick={toggle} />
            </Tooltip>
          </Space>
        </Header>

        <Content
          style={{
            padding: isMobile ? "16px 12px 40px" : "28px 32px 56px",
            maxWidth: 1500,
            margin: "0 auto",
            width: "100%",
          }}
        >
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
}
