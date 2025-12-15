import { useState } from "react";
import "./health_form.css";
import usePredict from "../hooks/usePredict";
import type {
  PatientInput,
  PatientInputDirect,
  PredictionResponse,
  PredictionResponseDirect,
} from "../service/api";

const SYMPTOM_OPTIONS = [
  "Fatigue",
  "Sore throat",
  "Body ache",
  "Shortness of breath",
  "Runny nose",
  "Headache",
  "Cough",
  "Fever",
] as const;

type HealthFormPageProps = {
  username?: string;
  onDiagnosisComplete: (
    result: PredictionResponse | PredictionResponseDirect
  ) => void;
};

export default function HealthFormPage({
  username,
  onDiagnosisComplete,
}: HealthFormPageProps) {
  const [formType, setFormType] = useState<"description" | "individual">(
    "description"
  );
  const [form, setForm] = useState({
    gender: "",
    age: "",
    symptom: "",
    symptom1: "",
    symptom2: "",
    symptom3: "",
    heartRate: "",
    temperature: "",
    oxygenSaturation: "",
    systole: "",
    diastole: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { predictSentence, predictDirect } = usePredict();

  const handleChange =
    (field: keyof typeof form) =>
    (
      event: React.ChangeEvent<
        HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
      >
    ) => {
      setForm((prev) => ({ ...prev, [field]: event.target.value }));
      setErrors((prev) => ({ ...prev, [field]: "" }));
    };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    const age = Number(form.age);
    const heartRate = Number(form.heartRate);
    const temp = Number(form.temperature);
    const o2 = Number(form.oxygenSaturation);
    const systolic = Number(form.systole);
    const diastolic = Number(form.diastole);

    if (!form.age || isNaN(age)) {
      newErrors.age = "Age is required";
    } else if (age < 18 || age > 79) {
      newErrors.age = "Age must be between 18-79";
    } else if (!Number.isInteger(age)) {
      newErrors.age = "Age must be a whole number";
    }

    if (!form.gender || (form.gender !== "Male" && form.gender !== "Female")) {
      newErrors.gender = "Gender must be Male or Female";
    }

    if (formType === "description") {
      if (!form.symptom || form.symptom.trim() === "") {
        newErrors.symptom = "Symptom description is required";
      }
    } else {
      if (!form.symptom1 || !SYMPTOM_OPTIONS.includes(form.symptom1 as any)) {
        newErrors.symptom1 = "Please select a valid symptom";
      }
      if (!form.symptom2 || !SYMPTOM_OPTIONS.includes(form.symptom2 as any)) {
        newErrors.symptom2 = "Please select a valid symptom";
      }
      if (!form.symptom3 || !SYMPTOM_OPTIONS.includes(form.symptom3 as any)) {
        newErrors.symptom3 = "Please select a valid symptom";
      }
    }

    if (!form.heartRate || isNaN(heartRate)) {
      newErrors.heartRate = "Heart rate is required";
    } else if (heartRate < 60 || heartRate > 120) {
      newErrors.heartRate = "Heart rate must be 60-120 bpm";
    } else if (!Number.isInteger(heartRate)) {
      newErrors.heartRate = "Heart rate must be a whole number";
    }

    if (!form.temperature || isNaN(temp)) {
      newErrors.temperature = "Temperature is required";
    } else if (temp < 35.5 || temp > 40.0) {
      newErrors.temperature = "Temperature must be 35.5-40.0°C";
    }

    if (!form.oxygenSaturation || isNaN(o2)) {
      newErrors.oxygenSaturation = "O2 saturation is required";
    } else if (o2 < 90 || o2 > 99) {
      newErrors.oxygenSaturation = "O2 saturation must be 90-99%";
    } else if (!Number.isInteger(o2)) {
      newErrors.oxygenSaturation = "O2 saturation must be a whole number";
    }

    if (!form.systole || isNaN(systolic)) {
      newErrors.systole = "Systolic is required";
    } else if (systolic < 90 || systolic > 180) {
      newErrors.systole = "Systolic must be 90-180";
    } else if (!Number.isInteger(systolic)) {
      newErrors.systole = "Systolic must be a whole number";
    }

    if (!form.diastole || isNaN(diastolic)) {
      newErrors.diastole = "Diastolic is required";
    } else if (diastolic < 60 || diastolic > 120) {
      newErrors.diastole = "Diastolic must be 60-120";
    } else if (!Number.isInteger(diastolic)) {
      newErrors.diastole = "Diastolic must be a whole number";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!validateForm()) return;

    try {
      const baseData = {
        "Oxygen_Saturation_%": Number(form.oxygenSaturation),
        Body_Temperature_C: Number(form.temperature),
        Heart_Rate_bpm: Number(form.heartRate),
        Systolic: Number(form.systole),
        Diastolic: Number(form.diastole),
        Age: Number(form.age),
        Gender: form.gender as PatientInput.Gender,
      };

      let response;

      if (formType === "description") {
        const payload: PatientInput = {
          ...baseData,
          symptoms_description: form.symptom,
        };
        response = await predictSentence(payload);
      } else {
        const payload: PatientInputDirect = {
          ...baseData,
          Symptom_1: form.symptom1 as PatientInputDirect.Symptom_1,
          Symptom_2: form.symptom2 as PatientInputDirect.Symptom_2,
          Symptom_3: form.symptom3 as PatientInputDirect.Symptom_3,
        };
        response = await predictDirect(payload);
      }

      onDiagnosisComplete(response);
    } catch (error) {
      console.error("Form submission error:", error);
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
          <div className="hf-type-toggle hf-span-2">
            <button
              type="button"
              className={`hf-toggle-btn ${
                formType === "description" ? "active" : ""
              }`}
              onClick={() => setFormType("description")}
            >
              Description
            </button>
            <button
              type="button"
              className={`hf-toggle-btn ${
                formType === "individual" ? "active" : ""
              }`}
              onClick={() => setFormType("individual")}
            >
              Individual Symptoms
            </button>
          </div>

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
              min="18"
              max="79"
              inputMode="numeric"
              placeholder="e.g. 29"
              value={form.age}
              onChange={handleChange("age")}
              required
            />
            {errors.age && <span className="hf-error">{errors.age}</span>}
          </label>

          {formType === "description" ? (
            <label className="hf-field hf-span-2">
              <span className="hf-label">Symptom Description</span>
              <textarea
                rows={3}
                placeholder="Describe your main symptoms"
                value={form.symptom}
                onChange={handleChange("symptom")}
                required
              />
              {errors.symptom && (
                <span className="hf-error">{errors.symptom}</span>
              )}
            </label>
          ) : (
            <>
              <label className="hf-field">
                <span className="hf-label">Symptom 1</span>
                <select
                  value={form.symptom1}
                  onChange={handleChange("symptom1")}
                  required
                >
                  <option value="">Select symptom</option>
                  {SYMPTOM_OPTIONS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                {errors.symptom1 && (
                  <span className="hf-error">{errors.symptom1}</span>
                )}
              </label>
              <label className="hf-field">
                <span className="hf-label">Symptom 2</span>
                <select
                  value={form.symptom2}
                  onChange={handleChange("symptom2")}
                  required
                >
                  <option value="">Select symptom</option>
                  {SYMPTOM_OPTIONS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                {errors.symptom2 && (
                  <span className="hf-error">{errors.symptom2}</span>
                )}
              </label>
              <label className="hf-field">
                <span className="hf-label">Symptom 3</span>
                <select
                  value={form.symptom3}
                  onChange={handleChange("symptom3")}
                  required
                >
                  <option value="">Select symptom</option>
                  {SYMPTOM_OPTIONS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                {errors.symptom3 && (
                  <span className="hf-error">{errors.symptom3}</span>
                )}
              </label>
            </>
          )}

          <label className="hf-field">
            <span className="hf-label">Heart rate (bpm)</span>
            <input
              type="number"
              min="60"
              max="120"
              inputMode="numeric"
              placeholder="e.g. 72"
              value={form.heartRate}
              onChange={handleChange("heartRate")}
              required
            />
            {errors.heartRate && (
              <span className="hf-error">{errors.heartRate}</span>
            )}
          </label>

          <label className="hf-field">
            <span className="hf-label">Body temperature (°C)</span>
            <input
              type="number"
              step="0.1"
              min="35.5"
              max="40.0"
              inputMode="decimal"
              placeholder="e.g. 36.7"
              value={form.temperature}
              onChange={handleChange("temperature")}
              required
            />
            {errors.temperature && (
              <span className="hf-error">{errors.temperature}</span>
            )}
          </label>

          <label className="hf-field">
            <span className="hf-label">Oxygen saturation (%)</span>
            <input
              type="number"
              min="90"
              max="99"
              inputMode="numeric"
              placeholder="e.g. 98"
              value={form.oxygenSaturation}
              onChange={handleChange("oxygenSaturation")}
              required
            />
            {errors.oxygenSaturation && (
              <span className="hf-error">{errors.oxygenSaturation}</span>
            )}
          </label>

          <label className="hf-field hf-span-2">
            <span className="hf-label">
              Blood pressure (systole / diastole)
            </span>
            <div className="hf-bp-row">
              <div>
                <input
                  type="number"
                  min="90"
                  max="180"
                  inputMode="numeric"
                  placeholder="120"
                  value={form.systole}
                  onChange={handleChange("systole")}
                  required
                />
                {errors.systole && (
                  <span className="hf-error">{errors.systole}</span>
                )}
              </div>
              <span className="hf-bp-separator">/</span>
              <div>
                <input
                  type="number"
                  min="60"
                  max="120"
                  inputMode="numeric"
                  placeholder="80"
                  value={form.diastole}
                  onChange={handleChange("diastole")}
                  required
                />
                {errors.diastole && (
                  <span className="hf-error">{errors.diastole}</span>
                )}
              </div>
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
