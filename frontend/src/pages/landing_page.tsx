import { useState } from 'react'
import './landing_page.css'

type LandingPageProps = {
    onContinue: (username: string) => void
}

export default function LandingPage({ onContinue }: LandingPageProps) {
    const [username, setUsername] = useState('')
    const trimmed = username.trim()

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault()
        if (!trimmed) return
        onContinue(trimmed)
    }

    return (
        <main className="landing">
            <div className="backdrop">
                <span className="orb orb-a" aria-hidden />
                <span className="orb orb-b" aria-hidden />
                <span className="grid" aria-hidden />
            </div>

            <section className="panel">
                <h1 className="title">Start with your username</h1>

                <form className="field" onSubmit={handleSubmit}>
                    <div className="input-shell">
                        <input
                            id="username"
                            name="username"
                            type="text"
                            placeholder="username"
                            value={username}
                            onChange={(event) => setUsername(event.target.value)}
                            autoComplete="off"
                        />
                    </div>
                    <button className="primary-btn" type="submit" disabled={!trimmed}>
                        Continue
                    </button>
                </form>
            </section>
        </main>
    )
}