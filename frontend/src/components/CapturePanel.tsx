import { useEffect, useRef, useState } from "react";
import { Camera, CircleStop, Sparkles, Upload, Video } from "lucide-react";
import type { CameraAngle, Mode } from "../types";

interface Props {
  mode: Mode;
  angle: CameraAngle;
  useAiCoach: boolean;
  clip: File | null;
  clipUrl: string | null;
  onAngle: (angle: CameraAngle) => void;
  onAiCoach: (enabled: boolean) => void;
  onClip: (file: File) => void;
  onAnalyze: () => void;
}

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

export default function CapturePanel(props: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [cameraError, setCameraError] = useState("");
  const [facing, setFacing] = useState<"user" | "environment">("user");

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

  async function enableCamera(mode: "user" | "environment" = facing) {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      setFacing(mode);
      setCameraError("");
    } catch {
      setCameraError("Camera access was blocked. You can still upload a clip below.");
    }
  }

  async function record() {
    if (!streamRef.current || recording) return;
    for (const value of [3, 2, 1]) { setCountdown(value); await wait(700); }
    setCountdown(null);
    const mimeCandidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/mp4"];
    const mimeType = mimeCandidates.find((type) => MediaRecorder.isTypeSupported(type)) || "video/webm";
    const recorder = new MediaRecorder(streamRef.current, { mimeType });
    recorderRef.current = recorder;
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      const extension = recorder.mimeType.includes("mp4") ? "mp4" : "webm";
      props.onClip(new File([blob], `creaselab-${Date.now()}.${extension}`, { type: blob.type }));
      setRecording(false);
    };
    recorder.start(200);
    setRecording(true);
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    window.setTimeout(() => {
      window.clearInterval(timer);
      if (recorder.state === "recording") recorder.stop();
    }, 6000);
  }

  function stop() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  return (
    <section className="capture-shell" id="studio">
      <div className="section-kicker"><span>02</span> Capture studio</div>
      <div className="capture-grid">
        <div className="camera-card">
          <div className="camera-toolbar">
            <span className="live-pill"><i /> {recording ? `REC 00:0${Math.min(elapsed, 6)}` : cameraOn ? "CAMERA READY" : "PREVIEW"}</span>
            <div className="toolbar-actions">
              {cameraOn && (
                <button className="flip-button" onClick={() => enableCamera(facing === "user" ? "environment" : "user")} title="Switch camera">
                  Flip camera
                </button>
              )}
              <span className="mode-pill">{props.mode}</span>
            </div>
          </div>
          <div className="camera-viewport">
            {props.clipUrl && !cameraOn ? (
              <video src={props.clipUrl} controls playsInline />
            ) : (
              <video ref={videoRef} autoPlay muted playsInline className="camera-feed" />
            )}
            {!cameraOn && !props.clipUrl && (
              <button className="camera-empty" onClick={() => enableCamera()}>
                <span><Camera size={28} /></span>
                <strong>Enable camera</strong>
                <small>Nothing is uploaded until you choose Analyse</small>
              </button>
            )}
            {countdown !== null && <div className="countdown">{countdown}</div>}
            {cameraOn && !recording && <div className="frame-guide"><span>Keep your full body inside the frame</span></div>}
          </div>
          <div className="camera-actions">
            {cameraOn ? (
              <button className={`record-button ${recording ? "active" : ""}`} onClick={recording ? stop : record}>
                {recording ? <CircleStop size={20} /> : <span className="record-dot" />}
                {recording ? "Stop recording" : "Record 6 seconds"}
              </button>
            ) : (
              <button className="secondary-button" onClick={() => enableCamera()}><Video size={18} /> Use camera</button>
            )}
            <label className="upload-button">
              <Upload size={18} /> Upload clip
              <input type="file" accept="video/mp4,video/quicktime,video/webm,video/x-msvideo" onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) { streamRef.current?.getTracks().forEach((track) => track.stop()); setCameraOn(false); props.onClip(file); }
              }} />
            </label>
          </div>
          {cameraError && <p className="inline-error">{cameraError}</p>}
        </div>

        <aside className="setup-card">
          <div>
            <div className="setup-number">1</div>
            <h3>Frame the whole movement</h3>
            <p>Place the camera 4–6 metres away. Keep feet, hands, and follow-through visible.</p>
          </div>
          <div className="field">
            <label>Camera angle</label>
            <div className="segmented">
              {(["side_on", "front_on", "rear"] as CameraAngle[]).map((angle) => (
                <button key={angle} className={props.angle === angle ? "selected" : ""} onClick={() => props.onAngle(angle)}>
                  {angle.replace("_", " ")}
                </button>
              ))}
            </div>
          </div>
          <label className="ai-toggle">
            <input type="checkbox" checked={props.useAiCoach} onChange={(event) => props.onAiCoach(event.target.checked)} />
            <span className="toggle-track"><span /></span>
            <span><strong><Sparkles size={15} /> Enhanced AI coach</strong><small>When configured, shares 3 selected stills and metrics with OpenAI.</small></span>
          </label>
          <div className="clip-status">
            <span className={props.clip ? "ready" : ""}>{props.clip ? "Clip ready" : "No clip selected"}</span>
            {props.clip && <small>{props.clip.name} · {(props.clip.size / 1024 / 1024).toFixed(1)} MB</small>}
          </div>
          <button className="primary-button full" disabled={!props.clip} onClick={props.onAnalyze}>
            Analyse my action <Sparkles size={18} />
          </button>
          <p className="privacy-line">Clips are processed temporarily and not added to a training dataset.</p>
        </aside>
      </div>
    </section>
  );
}
