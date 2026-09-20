import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Eye, EyeOff, Gauge, Maximize, Repeat, SkipBack, SkipForward } from "lucide-react";
import type { Phase, TimelineFrame } from "../types";

const CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32], [15, 17], [15, 19], [16, 18], [16, 20]];

interface Props {
  src: string;
  frames: TimelineFrame[];
  phases: Phase[];
  fps: number;
  highlightJoints?: number[];
  onVideo?: (video: HTMLVideoElement) => void;
}

export default function VideoReplay({ src, frames, phases, fps, highlightJoints, onVideo }: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlay, setOverlay] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [looping, setLooping] = useState(false);
  const [paused, setPaused] = useState(true);
  const [durationMs, setDurationMs] = useState(3000);
  const [playbackError, setPlaybackError] = useState(false);

  // Automatic muted replay when the result arrives.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = true;
    video.play().catch(() => { setPaused(true); });
    return () => video.pause();
  }, []);

  // Pause instead of playing to the end, so the overlay stays on the action.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => {
      setPaused(video.paused);
      if (!looping) return;
      const end = phases[phases.length - 1];
      if (end && video.currentTime * 1000 >= end.timestamp_ms + 40) {
        const start = phases[0];
        video.currentTime = (start ? start.timestamp_ms : 0) / 1000;
      }
    };
    const onMeta = () => setDurationMs(video.duration * 1000 || 3000);
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("loadedmetadata", onMeta);
    return () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("loadedmetadata", onMeta);
    };
  }, [looping, phases]);

  // Synchronised pose overlay: canvas coordinates match the displayed video
  // rect exactly (letterboxing included), while drawing uses video pixels.
  const layoutCanvas = useCallback(() => {
    const stage = stageRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!stage || !video || !canvas) return;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
    const displayWidth = width * scale;
    const displayHeight = height * scale;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
    canvas.style.left = `${(stage.clientWidth - displayWidth) / 2}px`;
    canvas.style.top = `${(stage.clientHeight - displayHeight) / 2}px`;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    const video = videoRef.current;
    if (!stage || !video) return;
    onVideo?.(video);
    layoutCanvas();
    // The intrinsic video size is unknown until metadata arrives, so lay out
    // again then rather than relying on the resize observer alone.
    video.addEventListener("loadedmetadata", layoutCanvas);
    const observer = new ResizeObserver(layoutCanvas);
    observer.observe(stage);
    return () => {
      video.removeEventListener("loadedmetadata", layoutCanvas);
      observer.disconnect();
    };
  }, [layoutCanvas, onVideo]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    let animation = 0;
    const draw = () => {
      const width = video.videoWidth || 1280;
      const height = video.videoHeight || 720;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (overlay && !playbackError && frames.length) {
        const target = video.currentTime * 1000;
        let nearest = frames[0];
        for (const frame of frames) {
          if (Math.abs(frame.timestamp_ms - target) < Math.abs(nearest.timestamp_ms - target)) nearest = frame;
        }
        const points = new Map(nearest.landmarks.map(([index, x, y, visibility]) => [index, { x: x * width, y: y * height, visibility }]));
        const highlighted = new Set(highlightJoints ?? []);
        context.lineCap = "round";
        context.lineWidth = Math.max(3, width / 360);
        CONNECTIONS.forEach(([a, b]) => {
          const start = points.get(a), end = points.get(b);
          if (!start || !end || start.visibility < 0.4 || end.visibility < 0.4) return;
          const hot = highlighted.has(a) && highlighted.has(b);
          context.strokeStyle = hot ? "rgba(255, 214, 102, .98)" : "rgba(164, 255, 32, .9)";
          context.shadowColor = hot ? "rgba(255, 190, 40, .8)" : "rgba(164, 255, 32, .7)";
          context.shadowBlur = 12;
          context.beginPath();
          context.moveTo(start.x, start.y);
          context.lineTo(end.x, end.y);
          context.stroke();
        });
        points.forEach((point, index) => {
          if (point.visibility <= 0.4) return;
          const hot = highlighted.has(index);
          context.fillStyle = hot ? "#ffd666" : "#eaffcd";
          context.beginPath();
          context.arc(point.x, point.y, Math.max(4, width / 280) * (hot ? 1.6 : 1), 0, Math.PI * 2);
          context.fill();
        });
      }
      animation = requestAnimationFrame(draw);
    };
    animation = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animation);
  }, [frames, overlay, highlightJoints, playbackError]);

  function togglePlay() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => undefined);
    else video.pause();
  }

  function step(amount: number) {
    const video = videoRef.current;
    if (!video) return;
    const interval = 1 / (fps || 30);
    video.currentTime = Math.min(Math.max(0, video.currentTime + amount * interval), video.duration || 0);
  }

  function setPlayback(next: number) {
    setSpeed(next);
    if (videoRef.current) videoRef.current.playbackRate = next;
  }

  function seek(ms: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = ms / 1000;
    video.play().catch(() => undefined);
  }

  function fullscreen() {
    stageRef.current?.requestFullscreen?.().catch(() => undefined);
  }

  const lastPhase = phases[phases.length - 1];
  const trackEnd = Math.max(durationMs, lastPhase ? lastPhase.timestamp_ms : 3000, 1);

  return (
    <div className="replay-card">
      <div className="replay-stage" ref={stageRef}>
        <video ref={videoRef} src={src} controls playsInline preload="auto" onError={() => setPlaybackError(true)} hidden={playbackError} />
        <canvas ref={canvasRef} className="pose-canvas" />
        <div className="replay-badge">ANALYSIS REPLAY</div>
        {playbackError && (
          <div className="replay-fallback">
            <AlertTriangle size={22} />
            <strong>This clip cannot be replayed in your browser</strong>
            <p>
              The video uses a codec the browser will not decode (MPEG-4 Part 2 and HEVC are common culprits).
              The classification, phases, measurements and coaching below are still valid — re-export the clip as
              MP4/H.264, or record in the browser, to see the synchronised pose replay.
            </p>
          </div>
        )}
      </div>
      <div className="replay-controls">
        <button onClick={togglePlay} className="play-toggle" disabled={playbackError}>{paused ? "Play" : "Pause"}</button>
        <button onClick={() => step(-1)} title="Previous frame" disabled={playbackError}><SkipBack size={15} /> Frame</button>
        <button onClick={() => step(1)} title="Next frame" disabled={playbackError}><SkipForward size={15} /> Frame</button>
        <button onClick={() => setOverlay(!overlay)} className={overlay ? "active" : ""} title="Toggle pose overlay" disabled={playbackError}>
          {overlay ? <Eye size={16} /> : <EyeOff size={16} />} Pose
        </button>
        <div className="speed-control"><Gauge size={16} /> {[0.25, 0.5, 1].map((value) => <button key={value} className={speed === value ? "active" : ""} onClick={() => setPlayback(value)} disabled={playbackError}>{value}×</button>)}</div>
        <button onClick={() => setLooping(!looping)} className={looping ? "active" : ""} title="Loop the detected action phase" disabled={playbackError}><Repeat size={16} /> Loop</button>
        <button onClick={fullscreen} title="Fullscreen" disabled={playbackError}><Maximize size={16} /></button>
      </div>
      <div className="phase-track">
        <span className="track-line" />
        {phases.map((phase, index) => {
          const left = 8 + 84 * (phase.timestamp_ms / trackEnd);
          return (
            <button
              key={phase.id}
              className={index % 2 === 0 ? "above" : "below"}
              style={{ left: `${left}%` }}
              onClick={() => seek(phase.timestamp_ms)}
              title={`Seek to ${phase.label}`}
            >
              <i />
              <strong>{phase.label}</strong>
              <small>{(phase.timestamp_ms / 1000).toFixed(2)}s</small>
            </button>
          );
        })}
      </div>
    </div>
  );
}
