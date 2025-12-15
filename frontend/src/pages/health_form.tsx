import { useState } from "react";
import "./health_form.css";
import usePredict from "../hooks/usePredict";
import type { PatientInput, PredictionResponse } from "../service/api";

type HealthFormPageProps = {
  username?: string;
  onDiagnosisComplete: (result: PredictionResponse) => void;
};

export default function HealthFormPage({
  username,
  onDiagnosisComplete,
}: HealthFormPageProps) {
  const [form, setForm] = useState({
    gender: "",
    age: "",
    symptom: "",
    heartRate: "",
    temperature: "",
    oxygenSaturation: "",
    systole: "",
    diastole: "",
  });

  const { predictSentence } = usePredict();

  const handleChange =
    (field: keyof typeof form) =>
    (
      event: React.ChangeEvent<
        HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
      >
    ) => {
      setForm((prev) => ({ ...prev, [field]: event.target.value }));
    };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const response = await predictSentence({
        "Oxygen_Saturation_%": Number(form.oxygenSaturation),
        symptoms_description: form.symptom,
        Body_Temperature_C: Number(form.temperature),
        Heart_Rate_bpm: Number(form.heartRate),
        Systolic: Number(form.systole),
        Diastolic: Number(form.diastole),
        Age: Number(form.age),
        Gender: form.gender as PatientInput.Gender,
      });
      onDiagnosisComplete(response);
    } catch (error) {
      alert("There was an error submitting the form. Please try again.");
    }
  };

  return (
    <main className="health-form">
      <div className="hf-backdrop" aria-hidden>
        <span className="hf-orb orb-a" />
        <span className="hf-orb orb-b" />
        <span className="hf-grid" />
      </div>

      <section className="hf-card">
        <header className="hf-header">
          <p className="hf-eyebrow">Health intake</p>
          <h2 className="hf-greeting">
            {username ? `Hello ${username}` : "Hello there"}
          </h2>
          <h1 className="hf-title">Provide your details</h1>
        </header>

        <form className="hf-form" onSubmit={handleSubmit}>
          <fieldset className="hf-field hf-fieldset">
            <legend className="hf-label">Gender</legend>
            <div className="hf-radio-group">
              <label className="hf-radio">
                <input
                  type="radio"
                  name="gender"
                  value="Male"
                  checked={form.gender === "Male"}
                  onChange={handleChange("gender")}
                  required
                />
                <span>Male</span>
              </label>
              <label className="hf-radio">
                <input
                  type="radio"
                  name="gender"
                  value="Female"
                  checked={form.gender === "Female"}
                  onChange={handleChange("gender")}
                  required
                />
                <span>Female</span>
              </label>
            </div>
          </fieldset>

          <label className="hf-field">
            <span className="hf-label">Age</span>
            <input
              type="number"
              min="0"
              inputMode="numeric"
              placeholder="e.g. 29"
              value={form.age}
              onChange={handleChange("age")}
              required
            />
          </label>

          <label className="hf-field hf-span-2">
            <span className="hf-label">Symptom</span>
            <textarea
              rows={3}
              placeholder="Describe your main symptoms"
              value={form.symptom}
              onChange={handleChange("symptom")}
              required
            />
          </label>

          <label className="hf-field">
            <span className="hf-label">Heart rate (bpm)</span>
            <input
              type="number"
              min="0"
              inputMode="numeric"
              placeholder="e.g. 72"
              value={form.heartRate}
              onChange={handleChange("heartRate")}
            />
          </label>

          <label className="hf-field">
            <span className="hf-label">Body temperature (°C)</span>
            <input
              type="number"
              step="0.1"
              min="30"
              inputMode="decimal"
              placeholder="e.g. 36.7"
              value={form.temperature}
              onChange={handleChange("temperature")}
            />
          </label>

          <label className="hf-field">
            <span className="hf-label">Oxygen saturation (%)</span>
            <input
              type="number"
              min="0"
              max="100"
              inputMode="numeric"
              placeholder="e.g. 98"
              value={form.oxygenSaturation}
              onChange={handleChange("oxygenSaturation")}
            />
          </label>

          <label className="hf-field hf-span-2">
            <span className="hf-label">
              Blood pressure (systole / diastole)
            </span>
            <div className="hf-bp-row">
              <input
                type="number"
                min="50"
                inputMode="numeric"
                placeholder="120"
                value={form.systole}
                onChange={handleChange("systole")}
              />
              <span className="hf-bp-separator">/</span>
              <input
                type="number"
                min="30"
                inputMode="numeric"
                placeholder="80"
                value={form.diastole}
                onChange={handleChange("diastole")}
              />
            </div>
          </label>

          <div className="hf-actions hf-span-2">
            <button type="submit">Submit details</button>
          </div>
        </form>
      </section>
    </main>
  );
}
