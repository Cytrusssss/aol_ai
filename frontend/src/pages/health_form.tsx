import { useState } from 'react'
import './health_form.css'

type HealthFormPageProps = {
  username?: string
}

export default function HealthFormPage({ username }: HealthFormPageProps) {
  const [form, setForm] = useState({
    gender: '',
    age: '',
    symptom: '',
    heartRate: '',
    temperature: '',
    oxygenSaturation: '',
    systole: '',
    diastole: '',
  })

  const handleChange = (field: keyof typeof form) => (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>,
  ) => {
    setForm((prev) => ({ ...prev, [field]: event.target.value }))
  }

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    // Placeholder submit handler; integrate with backend as needed.
    console.log('Submitted form', form)
  }

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
          <h2 className="hf-greeting">{username ? `Hello ${username}` : 'Hello there'}</h2>
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
                  value="male"
                  checked={form.gender === 'male'}
                  onChange={handleChange('gender')}
                  required
                />
                <span>Male</span>
              </label>
              <label className="hf-radio">
                <input
                  type="radio"
                  name="gender"
                  value="female"
                  checked={form.gender === 'female'}
                  onChange={handleChange('gender')}
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
              onChange={handleChange('age')}
              required
            />
          </label>

          <label className="hf-field hf-span-2">
            <span className="hf-label">Symptom</span>
            <textarea
              rows={3}
              placeholder="Describe your main symptoms"
              value={form.symptom}
              onChange={handleChange('symptom')}
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
              onChange={handleChange('heartRate')}
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
              onChange={handleChange('temperature')}
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
              onChange={handleChange('oxygenSaturation')}
            />
          </label>

          <label className="hf-field hf-span-2">
            <span className="hf-label">Blood pressure (systole / diastole)</span>
            <div className="hf-bp-row">
              <input
                type="number"
                min="50"
                inputMode="numeric"
                placeholder="120"
                value={form.systole}
                onChange={handleChange('systole')}
              />
              <span className="hf-bp-separator">/</span>
              <input
                type="number"
                min="30"
                inputMode="numeric"
                placeholder="80"
                value={form.diastole}
                onChange={handleChange('diastole')}
              />
            </div>
          </label>

          <div className="hf-actions hf-span-2">
            <button type="submit">Submit details</button>
          </div>
        </form>
      </section>
    </main>
  )
}
