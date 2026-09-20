import { useEffect, useRef, useState } from "react";
import { Eye, EyeOff, Gauge } from "lucide-react";
import type { Phase, TimelineFrame } from "../types";

const CONNECTIONS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32],[15,17],[15,19],[16,18],[16,20]];

interface Props { src: string; frames: TimelineFrame[]; phases: Phase[]; onVideo?: (video: HTMLVideoElement) => void }

export default function VideoReplay({ src, frames, phases, onVideo }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlay, setOverlay] = useState(true);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    onVideo?.(video);
    let animation = 0;
    const draw = () => {
      const width = video.videoWidth || 1280, height = video.videoHeight || 720;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, width, height);
      if (overlay && frames.length) {
        const target = video.currentTime * 1000;
        let nearest = frames[0];
        for (const frame of frames) {
          if (Math.abs(frame.timestamp_ms - target) < Math.abs(nearest.timestamp_ms - target)) nearest = frame;
        }
        const points = new Map(nearest.landmarks.map(([index, x, y, visibility]) => [index, { x: x * width, y: y * height, visibility }]));
        context.lineCap = "round";
        context.lineWidth = Math.max(3, width / 360);
        context.strokeStyle = "rgba(164, 255, 32, .9)";
        context.shadowColor = "rgba(164, 255, 32, .7)";
        context.shadowBlur = 12;
        CONNECTIONS.forEach(([a, b]) => {
          const start = points.get(a), end = points.get(b);
          if (!start || !end || start.visibility < .4 || end.visibility < .4) return;
          context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
        });
        context.fillStyle = "#eaffcd";
        points.forEach((point) => { if (point.visibility > .4) { context.beginPath(); context.arc(point.x, point.y, Math.max(4, width / 280), 0, Math.PI * 2); context.fill(); } });
      }
      animation = requestAnimationFrame(draw);
    };
    animation = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animation);
  }, [frames, overlay, onVideo]);

  function setPlayback(next: number) {
    setSpeed(next);
    if (videoRef.current) videoRef.current.playbackRate = next;
  }

  function seek(ms: number) {
    if (videoRef.current) { videoRef.current.currentTime = ms / 1000; videoRef.current.play(); }
  }

  return (
    <div className="replay-card">
      <div className="replay-stage">
        <video ref={videoRef} src={src} controls playsInline />
        <canvas ref={canvasRef} className="pose-canvas" />
        <div className="replay-badge">ANALYSIS REPLAY</div>
      </div>
      <div className="replay-controls">
        <button onClick={() => setOverlay(!overlay)} className={overlay ? "active" : ""}>{overlay ? <Eye size={16}/> : <EyeOff size={16}/>} Pose</button>
        <div className="speed-control"><Gauge size={16} /> {[0.25, 0.5, 1].map((value) => <button key={value} className={speed === value ? "active" : ""} onClick={() => setPlayback(value)}>{value}×</button>)}</div>
      </div>
      <div className="phase-track">
        <span className="track-line" />
        {phases.map((phase, index) => (
          <button key={phase.id} style={{ left: `${10 + index * 40}%` }} onClick={() => seek(phase.timestamp_ms)}>
            <i /><strong>{phase.label}</strong><small>{(phase.timestamp_ms / 1000).toFixed(2)}s</small>
          </button>
        ))}
      </div>
    </div>
  );
}
