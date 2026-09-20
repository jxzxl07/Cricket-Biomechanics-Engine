import { useEffect, useState } from "react";
import { ArrowRight, BarChart3, BrainCircuit, ChevronRight, Crosshair, FlaskConical, ShieldCheck, Sparkles } from "lucide-react";
import { analyzeClip } from "./api";
import CapturePanel from "./components/CapturePanel";
import Results from "./components/Results";
import type { Analysis, CameraAngle, Mode } from "./types";

export default function App() {
  const [mode, setMode] = useState<Mode>("batting");
  const [angle, setAngle] = useState<CameraAngle>("side_on");
  const [useAiCoach, setUseAiCoach] = useState(false);
  const [clip, setClip] = useState<File | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => () => { if (clipUrl) URL.revokeObjectURL(clipUrl); }, [clipUrl]);

  function selectClip(file: File) {
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    setClip(file); setClipUrl(URL.createObjectURL(file)); setError("");
  }

  async function runAnalysis() {
    if (!clip) return;
    setLoading(true); setError("");
    try { setAnalysis(await analyzeClip(clip, mode, angle, useAiCoach)); window.scrollTo({ top: 0, behavior: "smooth" }); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Analysis failed"); }
    finally { setLoading(false); }
  }

  function reset() { setAnalysis(null); setClip(null); if (clipUrl) URL.revokeObjectURL(clipUrl); setClipUrl(null); setError(""); }

  if (analysis && clipUrl) return <><Nav /><Results analysis={analysis} clipUrl={clipUrl} onReset={reset} /><Footer /></>;

  return (
    <div className="app-shell">
      <Nav />
      <main>
        <section className="hero container">
          <div className="hero-copy">
            <div className="eyebrow"><Sparkles size={14}/> Cricket movement intelligence</div>
            <h1>See the action.<br/><em>Own the next one.</em></h1>
            <p>Record a batting shot or bowling action. CreaseLab classifies the movement, maps the key moments, and turns them into coaching cues you can actually use.</p>
            <a href="#choose" className="primary-button">Start an analysis <ArrowRight size={18}/></a>
            <div className="trust-row"><span><ShieldCheck size={15}/> No clip storage</span><span><Crosshair size={15}/> Pose-synchronised replay</span><span><BrainCircuit size={15}/> Evidence-led coaching</span></div>
          </div>
          <div className="hero-visual" aria-label="Abstract cricket motion graphic">
            <div className="orbit orbit-one"/><div className="orbit orbit-two"/>
            <div className="crease-line"/><div className="ball-glow"/>
            <div className="visual-stat stat-one"><small>PEAK PHASE</small><strong>0.84s</strong></div>
            <div className="visual-stat stat-two"><small>POSE COVERAGE</small><strong>96%</strong></div>
            <div className="motion-figure"><i/><i/><i/><i/><i/><i/><i/></div>
          </div>
        </section>

        <section className="mode-section container" id="choose">
          <div className="section-kicker"><span>01</span> Choose your discipline</div>
          <div className="section-heading"><h2>What are we analysing?</h2><p>Pick the action first—the models and movement signals are specific to each discipline.</p></div>
          <div className="mode-grid">
            <button className={`mode-card batting ${mode === "batting" ? "selected" : ""}`} onClick={() => setMode("batting")}>
              <span className="mode-number">01</span><div className="mode-art bat-art"><i/><i/></div>
              <div><span className="mode-label">BATTING</span><h3>Read the shot</h3><p>10 shot families, hand-speed timing, balance, rotation, and finish shape.</p><strong>Analyse batting <ChevronRight size={18}/></strong></div>
            </button>
            <button className={`mode-card bowling ${mode === "bowling" ? "selected" : ""}`} onClick={() => setMode("bowling")}>
              <span className="mode-number">02</span><div className="mode-art bowl-art"><i/><i/><i/></div>
              <div><span className="mode-label">BOWLING · EXPERIMENTAL</span><h3>Map the action</h3><p>Broad pace/spin family, release position, arm path, rotation, and follow-through.</p><strong>Analyse bowling <ChevronRight size={18}/></strong></div>
            </button>
          </div>
        </section>

        <div className="container"><CapturePanel mode={mode} angle={angle} useAiCoach={useAiCoach} clip={clip} clipUrl={clipUrl} onAngle={setAngle} onAiCoach={setUseAiCoach} onClip={selectClip} onAnalyze={runAnalysis}/>{error && <div className="error-banner">{error}</div>}</div>

        <section className="how-section container">
          <div className="section-kicker">Built for useful honesty</div><h2>More than a label.</h2>
          <div className="how-grid">
            <article><Crosshair/><span>01</span><h3>Classify</h3><p>A temporal video model reads the complete batting action—not a single frozen pose.</p></article>
            <article><BarChart3/><span>02</span><h3>Measure</h3><p>Pose signals reveal timing, ranges, release or peak speed, and capture quality.</p></article>
            <article><BrainCircuit/><span>03</span><h3>Improve</h3><p>Every coaching cue cites a visible phase or measured signal, with limitations made clear.</p></article>
          </div>
        </section>

        <section className="method-section container" id="method">
          <div><ShieldCheck size={22}/><span><strong>Private by design</strong>Clips are processed in a temporary directory and deleted. Nothing is stored, shared, or used for training.</span></div>
          <div><FlaskConical size={22}/><span><strong>Honest about the models</strong>Both action models are trained on this project's own clips and scored 29% (batting) and 57% (bowling) when a whole recording session is held out — so every label is marked experimental and the measured number is shown beside it.</span></div>
          <div><Crosshair size={22}/><span><strong>No verdicts, no diagnosis</strong>CreaseLab never judges bowling legality, estimates ball speed, or diagnoses injury. It measures visible movement and nothing more.</span></div>
        </section>
      </main>
      <Footer />
      {loading && <div className="loading-screen"><div className="loader-orbit"><i/></div><strong>Reading the movement</strong><p>Classifying the clip, mapping 33 pose points, and building your replay…</p></div>}
    </div>
  );
}

function Nav() {
  return <nav className="nav"><div className="container nav-inner"><a href="/" className="brand"><span>CL</span> CREASELAB</a><div className="nav-links"><a href="#choose">Analyse</a><a href="#method">Method</a></div><span className="beta-pill">PUBLIC BETA</span></div></nav>;
}

function Footer() {
  return <footer><div className="container"><div className="brand"><span>CL</span> CREASELAB</div><p>Cricket movement intelligence for practice—not officiating, diagnosis, or certainty.</p><small>© 2026 CreaseLab</small></div></footer>;
}
