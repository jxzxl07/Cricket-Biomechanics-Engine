import { useMemo, useRef, useState } from "react";
import { Activity, AlertTriangle, BrainCircuit, CheckCircle2, FlaskConical, Info, RotateCcw, ShieldCheck, Sparkles, Target } from "lucide-react";
import type { Analysis, Metric } from "../types";
import VideoReplay from "./VideoReplay";

interface Props { analysis: Analysis; clipUrl: string; onReset: () => void }

function prettyLabel(label: string) { return label.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }

const BATTING_JOINTS: Record<string, number[]> = {
  peak_hand_speed: [15, 16],
  peak_speed_timing: [15, 16],
  shoulder_rotation_range: [11, 12],
  max_knee_bend: [25, 26],
  head_drop: [0],
  final_hand_height: [15, 16],
};

const BOWLING_JOINTS: Record<string, number[]> = {
  peak_wrist_speed: [15, 16],
  release_height: [15, 16, 23, 24],
  release_forward_reach: [11, 12, 15, 16],
  shoulder_rotation_range: [11, 12],
  torso_lean_at_release: [11, 12, 23, 24],
  elbow_angle_at_release: [13, 14, 15, 16],
};

function jointsFor(analysis: Analysis, metric: Metric): number[] {
  const table = analysis.mode === "batting" ? BATTING_JOINTS : BOWLING_JOINTS;
  let joints = table[metric.key] ?? [];
  const arm = analysis.features.detected_bowling_arm;
  if (arm === "left") joints = joints.map((joint) => (joint === 14 ? 13 : joint === 16 ? 15 : joint === 12 ? 11 : joint));
  if (arm === "right") joints = joints.map((joint) => (joint === 13 ? 14 : joint === 15 ? 16 : joint === 11 ? 12 : joint));
  return joints;
}

export default function Results({ analysis, clipUrl, onReset }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [highlight, setHighlight] = useState<number[]>([]);
  const classification = analysis.classification;
  const model = classification.model;
  const confidence = Math.round(classification.confidence * 100);
  const benchmark = model.benchmark;
  const top1Bench = benchmark.top1_accuracy !== undefined ? Math.round(benchmark.top1_accuracy * 100) : null;
  const ranked = useMemo(
    () => Object.entries(classification.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3),
    [classification.probabilities]
  );
  const seek = (ms: number | null) => {
    if (ms !== null && videoRef.current) { videoRef.current.currentTime = ms / 1000; videoRef.current.play().catch(() => undefined); window.scrollTo({ top: 100, behavior: "smooth" }); }
  };
  const needsBetterClip = analysis.status === "needs_better_clip";

  return (
    <main className="results-page">
      <header className="results-header container">
        <div><div className="section-kicker"><span>03</span> Performance report</div><h1>Your action, <em>decoded.</em></h1></div>
        <button className="secondary-button" onClick={onReset}><RotateCcw size={17} /> Analyse another</button>
      </header>

      {needsBetterClip && (
        <section className="rerecord-banner container">
          <AlertTriangle size={20} />
          <div>
            <strong>The clip needs a cleaner recording before results can be trusted.</strong>
            <ul className="rerecord-reasons">
              {analysis.quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}
            </ul>
            <ul>
              <li>Place the camera 4–6 m away at torso height, side-on to the action.</li>
              <li>Keep the whole body in frame from the start of the movement to the end of the follow-through.</li>
              <li>Make sure you are the only person in the shot, in even light.</li>
            </ul>
          </div>
          <button className="primary-button" onClick={onReset}><RotateCcw size={16} /> Re-record</button>
        </section>
      )}

      {!needsBetterClip && analysis.warnings.length > 0 && (
        <section className="warnings-banner container">
          <AlertTriangle size={17} />
          <div>{analysis.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>
        </section>
      )}

      <section className="result-hero container">
        <VideoReplay
          src={clipUrl}
          frames={analysis.timeline.frames}
          phases={analysis.phases}
          fps={analysis.timeline.fps}
          highlightJoints={highlight}
          onVideo={(video) => { videoRef.current = video; }}
        />
        <aside className="classification-card">
          <div className="classification-topline">
            <span>{analysis.mode} classification</span>
            <span className={model.experimental ? "tag experimental" : "tag"}>{model.experimental ? <><FlaskConical size={12} /> EXPERIMENTAL</> : "ACTIVE"}</span>
          </div>
          <div className="class-label">{classification.display_label}</div>
          {classification.unknown ? (
            <div className="unknown-note">
              <strong>No label this time.</strong>
              <p>The model was not confident enough to name the action. The replay, phases and measurements below are still valid.</p>
            </div>
          ) : (
            <>
              <div className="confidence-row">
                <div className="confidence-ring" style={{ "--score": `${Math.min(confidence, 65) * 3.6}deg` } as React.CSSProperties}><span>{confidence}%</span></div>
                <div><strong>{model.experimental ? "Model score (capped)" : "Model confidence"}</strong><small>{model.experimental ? `Reported at most ${Math.round(model.confidence_cap * 100)}% because the model is experimental` : classification.low_confidence ? "Interpret cautiously" : "Strong model match"}</small></div>
              </div>
              <div className="probability-heading">Raw model scores — uncalibrated</div>
              <div className="probability-list">
                {ranked.map(([label, score]) => <div key={label}><span>{prettyLabel(label)}</span><i><b style={{ width: `${Math.min(score, 1) * 100}%` }} /></i><strong>{Math.round(score * 100)}%</strong></div>)}
              </div>
              {model.experimental && (
                <div className="experiment-note">
                  <FlaskConical size={18} />
                  <span><strong>Experimental model</strong>
                    {top1Bench !== null ? <>On this project's own phone clips this model named the right shot {top1Bench}% of the time (top-2: {benchmark.top2_accuracy !== undefined ? Math.round(benchmark.top2_accuracy * 100) : "—"}%). Treat the label as a weak hint, not a verdict.</> : "No independent accuracy measurement exists for this model. Treat the label as a weak hint."}
                  </span>
                </div>
              )}
              {!model.experimental && <p className="model-note"><Info size={15} />{model.note}</p>}
            </>
          )}
          <div className="quality-score"><span><ShieldCheck size={18} /> Capture quality</span><strong>{analysis.quality.score}/100</strong></div>
          <div className="quality-checks">
            {analysis.quality.checks.map((check) => (
              <div key={check.id} className={check.passed ? "passed" : "failed"} title={check.detail}>
                {check.passed ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                <span>{check.label}</span>
              </div>
            ))}
          </div>
        </aside>
      </section>

      <section className="metrics-section container">
        <div className="section-heading">
          <div><div className="section-kicker"><span>LIVE</span> Movement signals</div><h2>The numbers behind the action</h2></div>
          <p>Click any card to jump the replay to its measured moment. The skeleton lights up on the joints that produced the number.</p>
        </div>
        <div className="metric-grid">
          {analysis.metrics.map((metric, index) => (
            <button
              className="metric-card"
              key={metric.key}
              onClick={() => seek(metric.timestamp_ms)}
              onMouseEnter={() => setHighlight(jointsFor(analysis, metric))}
              onMouseLeave={() => setHighlight([])}
              onFocus={() => setHighlight(jointsFor(analysis, metric))}
              onBlur={() => setHighlight([])}
            >
              <span className="metric-index">0{index + 1}</span>
              <span className="metric-value">{metric.value ?? "—"}<small>{metric.unit}</small></span>
              <strong>{metric.label}</strong><p>{metric.description}</p>
            </button>
          ))}
        </div>
        <div className="timing-strip">
          <Activity size={15} /> Analysis took {(analysis.timings.total_ms / 1000).toFixed(1)}s · pose {(analysis.timings.pose_ms / 1000).toFixed(1)}s · model {(analysis.timings.classification_ms / 1000).toFixed(1)}s
        </div>
      </section>

      <section className="coach-section">
        <div className="container coach-grid">
          <div className="coach-intro">
            <span className="coach-icon"><BrainCircuit size={25} /></span>
            <div className="section-kicker">{analysis.coach.enhanced ? "AI coach" : "Movement coach"}</div>
            <h2>A sharper next session starts here.</h2>
            <p>{analysis.coach.summary}</p>
            <div className="provider-pill"><Sparkles size={14} />{analysis.coach.enhanced ? "OpenAI-enhanced" : "Metric-grounded"}</div>
            <div className="uncertainty-box"><Info size={16} /><p>{analysis.coach.uncertainty}</p></div>
          </div>
          <div className="coach-feedback">
            {!!analysis.coach.strengths.length && <div className="feedback-block strengths"><h3><CheckCircle2 size={19} /> What looked strong</h3>{analysis.coach.strengths.map((item) => <p key={item}>{item}</p>)}</div>}
            <div className="feedback-block"><h3><Target size={19} /> Focus next</h3>{analysis.coach.improvements.map((item) => <article key={item.title}><strong>{item.title}</strong><small>{item.evidence}</small><p>"{item.cue}"</p></article>)}</div>
            <div className="drill-card"><span>YOUR NEXT DRILL</span><strong>{analysis.coach.drill}</strong></div>
          </div>
        </div>
      </section>

      <section className="method-section container">
        <div><ShieldCheck size={22} /><span><strong>Private by design</strong>{analysis.privacy}</span></div>
        <div><Info size={22} /><span><strong>Honest by design</strong>{analysis.coach.disclaimer}</span></div>
        {analysis.quality.notes.map((note) => <div key={note}><Activity size={22} /><span><strong>Capture note</strong>{note}</span></div>)}
      </section>
    </main>
  );
}
