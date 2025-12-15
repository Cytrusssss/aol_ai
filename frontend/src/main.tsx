import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import HealthFormPage from "./pages/health_form";
import LandingPage from "./pages/landing_page";
import DiagnosisResultPage from "./pages/diagnosis_result";
import { OpenAPI, type PredictionResponse } from "./service/api";

// Configure the API base URL
OpenAPI.BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function AppRouter() {
  const [username, setUsername] = useState<string | null>(() =>
    sessionStorage.getItem("username")
  );
  const [path, setPath] = useState(() => window.location.pathname || "/");
  const [diagnosisResult, setDiagnosisResult] =
    useState<PredictionResponse | null>(null);

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname || "/");
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const go = (nextPath: string) => {
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
      setPath(nextPath);
    }
  };

  const handleContinue = (name: string) => {
    setUsername(name);
    sessionStorage.setItem("username", name);
    go("/health");
  };

  const handleDiagnosisComplete = (result: PredictionResponse) => {
    setDiagnosisResult(result);
    go("/result");
  };

  const handleStartOver = () => {
    setDiagnosisResult(null);
    go("/");
  };

  if (path === "/result") {
    if (!diagnosisResult) {
      go("/");
      return <LandingPage onContinue={handleContinue} />;
    }
    return (
      <DiagnosisResultPage
        username={username || undefined}
        result={diagnosisResult}
        onStartOver={handleStartOver}
      />
    );
  }

  if (path === "/health") {
    if (!username) {
      go("/");
      return <LandingPage onContinue={handleContinue} />;
    }
    return (
      <HealthFormPage
        username={username}
        onDiagnosisComplete={handleDiagnosisComplete}
      />
    );
  }

  return <LandingPage onContinue={handleContinue} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AppRouter />
  </StrictMode>
);
