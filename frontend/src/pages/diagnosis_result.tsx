import { useState } from "react";
import "./diagnosis_result.css";
import type {
  PredictionResponse,
  PredictionResponseDirect,
} from "../service/api";

type DiagnosisResultPageProps = {
  username?: string;
  result: PredictionResponse | PredictionResponseDirect;
  onStartOver: () => void;
};

export default function DiagnosisResultPage({
  username,
  result,
  onStartOver,
}: DiagnosisResultPageProps) {
  const [showDetails, setShowDetails] = useState(false);

  const confidencePercent = (result.confidence * 100).toFixed(1);

  // Type guard to check if result has extracted symptoms
  const hasExtractedSymptoms = (
    res: PredictionResponse | PredictionResponseDirect
  ): res is PredictionResponse => {
    return "extracted_symptoms" in res;
  };

  return (
    <main className="diagnosis-result">
      <div className="dr-backdrop" aria-hidden>
        <span className="dr-orb orb-a" />
        <span className="dr-orb orb-b" />
        <span className="dr-grid" />
      </div>

      <section className="dr-card">
        <header className="dr-header">
          <p className="dr-eyebrow">Diagnosis complete</p>
          {username && <h2 className="dr-greeting">Results for {username}</h2>}
          <h1 className="dr-title">Your Diagnosis</h1>
        </header>

        <div className="dr-diagnosis-box">
          <div className="dr-diagnosis-label">Predicted Condition</div>
          <div className="dr-diagnosis-name">{result.predicted_diagnosis}</div>
          <div className="dr-confidence">
            <span className="dr-confidence-label">Confidence:</span>
            <span className="dr-confidence-value">{confidencePercent}%</span>
          </div>
        </div>

        <div className="dr-toggle-section">
          <button
            className="dr-toggle-btn"
            onClick={() => setShowDetails(!showDetails)}
            aria-expanded={showDetails}
          >
            <span>{showDetails ? "Hide" : "Show"} detailed response</span>
            <span className={`dr-chevron ${showDetails ? "open" : ""}`}>▼</span>
          </button>

          {showDetails && (
            <div className="dr-details">
              {hasExtractedSymptoms(result) && (
                <div className="dr-detail-block">
                  <h3 className="dr-detail-title">Extracted Symptoms</h3>
                  <div className="dr-symptom-list">
                    {result.extracted_symptoms.map((symptom, index) => (
                      <div key={index} className="dr-symptom-item">
                        <span className="dr-symptom-name">{symptom}</span>
                        {result.extraction_scores[symptom] && (
                          <span className="dr-symptom-score">
                            {(result.extraction_scores[symptom] * 100).toFixed(
                              0
                            )}
                            %
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="dr-detail-block">
                <h3 className="dr-detail-title">All Probabilities</h3>
                <div className="dr-probability-list">
                  {Object.entries(result.all_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([diagnosis, probability]) => (
                      <div key={diagnosis} className="dr-probability-item">
                        <span className="dr-prob-name">{diagnosis}</span>
                        <div className="dr-prob-bar-container">
                          <div
                            className="dr-prob-bar"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                        <span className="dr-prob-value">
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              <div className="dr-detail-block">
                <h3 className="dr-detail-title">Classification Details</h3>
                <div className="dr-meta-grid">
                  <div className="dr-meta-item">
                    <span className="dr-meta-label">Class Index:</span>
                    <span className="dr-meta-value">
                      {result.predicted_class_index}
                    </span>
                  </div>
                  <div className="dr-meta-item">
                    <span className="dr-meta-label">Confidence Score:</span>
                    <span className="dr-meta-value">
                      {result.confidence.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="dr-actions">
          <button className="dr-start-over-btn" onClick={onStartOver}>
            Start over
          </button>
        </div>
      </section>
    </main>
  );
}
