import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ConfigProvider, theme as antTheme, Spin, App as AntApp } from "antd";
import { ThemeModeProvider, useThemeMode } from "./context/ThemeContext";
import { BackendProvider } from "./context/BackendContext";
import BackendGate from "./components/BackendGate";
import Layout from "./components/Layout";

// Classify is the landing page and the thing most visitors came for, so it is
// the one route that loads eagerly. The rest — especially the recharts-heavy
// Metrics and Insights pages — are split out so a visitor on a slow connection
// isn't downloading the whole app before they can classify anything.
import Predict from "./pages/Predict";

const Overview = lazy(() => import("./pages/Overview"));
const Retrain = lazy(() => import("./pages/Retrain"));
const Insights = lazy(() => import("./pages/Insights"));
const Metrics = lazy(() => import("./pages/Metrics"));
const System = lazy(() => import("./pages/System"));
const About = lazy(() => import("./pages/About"));

function RouteFallback() {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh" }}>
      <Spin size="large" />
    </div>
  );
}

function Themed({ children }: { children: ReactNode }) {
  const { mode } = useThemeMode();
  return (
    <ConfigProvider
      theme={{
        algorithm: mode === "dark" ? antTheme.darkAlgorithm : antTheme.defaultAlgorithm,
        token: {
          colorPrimary: "#4338ca",
          colorLink: "#4338ca",
          borderRadius: 8,
          fontFamily: "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
          colorBgLayout: mode === "dark" ? "#141414" : "#f6f7fb",
        },
        components: {
          Menu: {
            itemSelectedBg: mode === "dark" ? "#221f4d" : "#eceafb",
            itemSelectedColor: "#4338ca",
          },
        },
      }}
    >
      <AntApp>{children}</AntApp>
    </ConfigProvider>
  );
}

const ROUTES: [string, React.ComponentType][] = [
  ["/overview", Overview],
  ["/training", Retrain],
  ["/dataset", Insights],
  ["/evaluation", Metrics],
  ["/status", System],
  ["/about", About],
];

export default function App() {
  return (
    <ThemeModeProvider>
      <Themed>
        <BackendProvider>
          <BackendGate>
            <BrowserRouter>
              <Routes>
                <Route element={<Layout />}>
                  <Route index element={<Predict />} />
                  {ROUTES.map(([path, Page]) => (
                    <Route
                      key={path}
                      path={path}
                      element={
                        <Suspense fallback={<RouteFallback />}>
                          <Page />
                        </Suspense>
                      }
                    />
                  ))}
                  {/* Any unknown client path lands on Classify rather than a
                      blank screen — the server hands every non-API path to
                      index.html, so this is where those arrive. /classify is
                      listed too, so an old link to it still works. */}
                  <Route path="/classify" element={<Predict />} />
                  <Route path="*" element={<Predict />} />
                </Route>
              </Routes>
            </BrowserRouter>
          </BackendGate>
        </BackendProvider>
      </Themed>
    </ThemeModeProvider>
  );
}
