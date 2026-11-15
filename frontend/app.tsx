import { useState, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type IncidentResult = {
  incident_id: string;
  analysis_source: "gemini" | "fallback";
  ai_analysis: {
    likely_cause: string;
    investigation_steps: string[];
    immediate_action: string;
  };
  retrieval_matches: { id: string; title: string; similarity: number }[];
};

function App() {
  const [result, setResult] = useState<IncidentResult | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function analyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    const form = new FormData(event.currentTarget);
    const apiKey = String(form.get("key")).trim();
    const incident = {
      service: String(form.get("service")).trim(),
      alert: String(form.get("alert")).trim(),
      severity: String(form.get("severity")),
      description: String(form.get("description")).trim(),
    };
    setResult(null);
    setError("");
    if (!apiKey || !incident.service || !incident.alert || !incident.description) {
      setError("Fill in every field before submitting.");
      return;
    }

    setPending(true);
    try {
      const response = await fetch("/incident", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify(incident),
      });
      if (!response.ok) {
        const detail: { detail?: unknown } = await response.json().catch(() => ({}));
        throw new Error(typeof detail.detail === "string"
          ? detail.detail
          : `Analysis failed (HTTP ${response.status}).`);
      }
      const data: { incident: IncidentResult } = await response.json();
      setResult(data.incident);
    } catch (failure) {
      setError(failure instanceof TypeError
        ? "Could not reach the Copilot API. Check that it is running on port 8000."
        : failure instanceof Error ? failure.message : "Analysis failed.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Incident Copilot</h1>
        <p>Submit a sample incident to retrieve related cases and generate an investigation plan.</p>
      </header>
      <div className="workspace">
        <form onSubmit={analyze}>
          <h2>New incident</h2>
          <label>Demo API key<input name="key" type="password" autoComplete="off" required /></label>
          <label>Service<input name="service" placeholder="payments" maxLength={120} required /></label>
          <label>Alert<input name="alert" placeholder="High checkout latency" maxLength={300} required /></label>
          <label>Severity<select name="severity" defaultValue="warning">
            <option value="info">Info</option><option value="warning">Warning</option><option value="critical">Critical</option>
          </select></label>
          <label>Description<textarea name="description" rows={4} maxLength={10000} required /></label>
          <button disabled={pending}>{pending ? "Analyzing…" : "Analyze and save"}</button>
          {error && <p role="alert" className="error">{error}</p>}
        </form>
        <section aria-busy={pending} aria-live="polite">
          <h2>Analysis</h2>
          {!result && <p>{pending ? "Retrieving context and analyzing the incident…" : "The investigation plan will appear here after submission."}</p>}
          {result && <>
            <p className="metadata">{result.incident_id} · {result.analysis_source === "gemini" ? "Gemini analysis" : "Fallback analysis"}</p>
            <h3>Likely cause</h3><p>{result.ai_analysis.likely_cause}</p>
            <h3>Immediate action</h3><p>{result.ai_analysis.immediate_action}</p>
            <h3>Investigation steps</h3>
            <ol>{result.ai_analysis.investigation_steps.map((step, index) => <li key={index}>{step}</li>)}</ol>
            <h3>Related knowledge</h3>
            {result.retrieval_matches.length
              ? <ul>{result.retrieval_matches.map(match => <li key={match.id}>{match.title} · Similarity {match.similarity.toFixed(2)}</li>)}</ul>
              : <p>No matches returned.</p>}
          </>}
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
