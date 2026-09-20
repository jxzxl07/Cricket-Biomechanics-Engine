import { useRef } from "react";
import { Activity, BrainCircuit, CheckCircle2, FlaskConical, Info, RotateCcw, ShieldCheck, Sparkles, Target } from "lucide-react";
import type { Analysis } from "../types";
import VideoReplay from "./VideoReplay";

interface Props { analysis: Analysis; clipUrl: string; onReset: () => void }

function prettyLabel(label: string) { return label.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }

export default function Results({ analysis, clipUrl, onReset }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const confidence = Math.round(analysis.classification.confidence * 100);
  const ranked = Object.entries(analysis.classification.probabilities).slice(0, 3);
  const seek = (ms: number | null) => { if (ms !== null && videoRef.current) { videoRef.current.currentTime = ms / 1000; videoRef.current.play(); window.scrollTo({ top: 100, behavior: "smooth" }); } };

  return (
    <main className="results-page">
      <header className="results-header container">
        <div><div className="section-kicker"><span>03</span> Performance report</div><h1>Your action, <em>decoded.</em></h1></div>
        <button className="secondary-button" onClick={onReset}><RotateCcw size={17}/> Analyse another</button>
      </header>

      <section className="result-hero container">
        <VideoReplay src={clipUrl} frames={analysis.timeline.frames} phases={analysis.phases} onVideo={(video) => { videoRef.current = video; }} />
        <aside className="classification-card">
          <div className="classification-topline"><span>{analysis.mode} classification</span><Activity size={17}/></div>
          <div className="class-label">{analysis.classification.display_label}</div>
          <div className="confidence-row">
            <div className="confidence-ring" style={{ "--score": `${confidence * 3.6}deg` } as React.CSSProperties}><span>{confidence}%</span></div>
            <div><strong>Model confidence</strong><small>{analysis.classification.low_confidence ? "Interpret cautiously" : "Strong model match"}</small></div>
          </div>
          <div className="probability-list">
            {ranked.map(([label, score]) => <div key={label}><span>{prettyLabel(label)}</span><i><b style={{ width: `${score * 100}%` }}/></i><strong>{Math.round(score * 100)}%</strong></div>)}
          </div>
          {analysis.classification.model.experimental && <div className="experiment-note"><FlaskConical size={18}/><span><strong>Experimental bowling model</strong>{analysis.classification.model.note}</span></div>}
          {!analysis.classification.model.experimental && <p className="model-note"><Info size={15}/>{analysis.classification.model.note}</p>}
          <div className="quality-score"><span><ShieldCheck size={18}/> Capture quality</span><strong>{analysis.quality.score}/100</strong></div>
        </aside>
      </section>

      <section className="metrics-section container">
        <div className="section-heading"><div><div className="section-kicker"><span>LIVE</span> Movement signals</div><h2>The numbers behind the action</h2></div><p>Click any card to jump to its measured moment.</p></div>
        <div className="metric-grid">
          {analysis.metrics.map((metric, index) => (
            <button className="metric-card" key={metric.key} onClick={() => seek(metric.timestamp_ms)}>
              <span className="metric-index">0{index + 1}</span>
              <span className="metric-value">{metric.value ?? "—"}<small>{metric.unit}</small></span>
              <strong>{metric.label}</strong><p>{metric.description}</p>
            </button>
          ))}
        </div>
      </section>

      <section className="coach-section">
        <div className="container coach-grid">
          <div className="coach-intro">
            <span className="coach-icon"><BrainCircuit size={25}/></span>
            <div className="section-kicker">{analysis.coach.enhanced ? "AI coach" : "Movement coach"}</div>
            <h2>A sharper next session starts here.</h2>
            <p>{analysis.coach.summary}</p>
            <div className="provider-pill"><Sparkles size={14}/>{analysis.coach.enhanced ? "OpenAI-enhanced" : "Metric-grounded"}</div>
          </div>
          <div className="coach-feedback">
            {!!analysis.coach.strengths.length && <div className="feedback-block strengths"><h3><CheckCircle2 size={19}/> What looked strong</h3>{analysis.coach.strengths.map((item) => <p key={item}>{item}</p>)}</div>}
            <div className="feedback-block"><h3><Target size={19}/> Focus next</h3>{analysis.coach.improvements.map((item) => <article key={item.title}><strong>{item.title}</strong><small>{item.evidence}</small><p>“{item.cue}”</p></article>)}</div>
            <div className="drill-card"><span>YOUR NEXT DRILL</span><strong>{analysis.coach.drill}</strong></div>
          </div>
        </div>
      </section>

      <section className="method-section container">
        <div><ShieldCheck size={22}/><span><strong>Private by design</strong>{analysis.privacy}</span></div>
        <div><Info size={22}/><span><strong>Honest by design</strong>{analysis.coach.disclaimer}</span></div>
        {analysis.quality.notes.map((note) => <div key={note}><Activity size={22}/><span><strong>Capture note</strong>{note}</span></div>)}
      </section>
    </main>
  );
}
