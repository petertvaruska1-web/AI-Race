import { useEffect, useMemo, useRef } from "react";
import { type OrbInput, type OrbParams, orbParams } from "./params.ts";

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  color: string;
  /** Dot radius: customers are bigger than data. */
  size: number;
};

const MAX_PARTICLES = 240;

function canDraw(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  try {
    return canvas.getContext("2d");
  } catch {
    return null;
  }
}

function reducedMotion(): boolean {
  return (
    typeof window.matchMedia !== "function" ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** Picks a color by share (shares add up to 1). */
function byShare(colors: string[], shares: number[]): string {
  let r = Math.random();
  for (let i = 0; i < colors.length; i++) {
    r -= shares[i] ?? 0;
    if (r <= 0) return colors[i] ?? "#ffffff";
  }
  return colors.at(-1) ?? "#ffffff";
}

function alpha(hex: string, a: number): string {
  const n = Number.parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${Math.max(0, Math.min(1, a))})`;
}

/** One frame: glow, rings, core, sparks and incoming data. `t` is seconds since the start. */
function drawFrame(
  ctx: CanvasRenderingContext2D,
  size: number,
  p: OrbParams,
  t: number,
  particles: Particle[],
  dim: number,
): void {
  const c = size / 2;
  const max = size * 0.3;
  const breathe = 1 + 0.05 * Math.sin(t * (2.4 - 1.6 * p.smoothness)) * (1 - 0.6 * p.smoothness);
  const r = Math.max(2, p.radius * max * breathe);
  const light = p.brightness * dim;
  const [primary = "#ffffff", , middle = primary] = p.colors;
  ctx.clearRect(0, 0, size, size);

  // The halo: softer and wider for a warmer AI.
  const halo = ctx.createRadialGradient(c, c, r * 0.3, c, c, r * (1.8 + 1.6 * p.glowSoftness));
  halo.addColorStop(0, alpha(primary, 0.45 * light));
  halo.addColorStop(1, alpha(primary, 0));
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, size, size);

  // Rings: one per skill it has, plus the words it has read; the last one fades in.
  const whole = Math.floor(p.rings);
  for (let i = 0; i < Math.ceil(p.rings); i++) {
    const a = i < whole ? 1 : p.rings - whole;
    const ring = r + max * (0.22 + 0.11 * i);
    const turn = t * (0.15 + 0.6 * p.swirl) * (0.6 + 0.25 * i) * (i % 2 === 0 ? 1 : -1);
    const tilt = 0.32 + 0.12 * Math.sin(t * (0.3 + p.turbulence) + i);
    ctx.save();
    ctx.translate(c, c);
    ctx.rotate(turn + i * 0.7);
    ctx.strokeStyle = alpha(p.colors[i % p.colors.length] ?? primary, 0.75 * a * dim);
    ctx.lineWidth = 1.4;
    const dashes = 2 + Math.round(10 * p.ringDetail);
    ctx.setLineDash([((Math.PI * 2 * ring) / dashes) * 0.7, ((Math.PI * 2 * ring) / dashes) * 0.3]);
    ctx.beginPath();
    ctx.ellipse(0, 0, ring, ring * tilt, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // The core.
  const core = ctx.createRadialGradient(c - r * 0.3, c - r * 0.3, r * 0.1, c, c, r);
  core.addColorStop(0, alpha("#ffffff", 0.95 * light));
  core.addColorStop(0.35, alpha(primary, 0.9 * light + 0.1));
  core.addColorStop(1, alpha(middle, 0.55 * light));
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.fill();

  for (const q of particles) {
    ctx.fillStyle = alpha(q.color, q.life * dim);
    ctx.beginPath();
    ctx.arc(q.x, q.y, q.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Moves particles on, adds new sparks (outward) and data (inward), drops the faded. */
function stepParticles(particles: Particle[], p: OrbParams, size: number, dt: number): Particle[] {
  const c = size / 2;
  const coreR = p.radius * size * 0.3;
  const live: Particle[] = [];
  for (const q of particles) {
    q.x += q.vx * dt;
    q.y += q.vy * dt;
    q.life -= dt * 0.8;
    const inward = (q.x - c) * q.vx + (q.y - c) * q.vy < 0;
    const reached = Math.hypot(q.x - c, q.y - c) < coreR;
    if (q.life > 0 && !(inward && reached)) live.push(q);
  }
  const sparks = p.sparkRate * 30 * dt;
  for (let i = 0; i < sparks || Math.random() < sparks - i; i++) {
    const a = Math.random() * Math.PI * 2;
    const v = 20 + 40 * Math.random();
    live.push({
      x: c + Math.cos(a) * coreR,
      y: c + Math.sin(a) * coreR,
      vx: Math.cos(a) * v,
      vy: Math.sin(a) * v,
      life: 1,
      color: p.colors[Math.floor(Math.random() * p.colors.length)] ?? "#ffffff",
      size: 1.6,
    });
    if (i > 5) break;
  }
  if (p.intake) {
    const n = p.intake.rate * 45 * dt;
    for (let i = 0; i < n || Math.random() < n - i; i++) {
      const a = Math.random() * Math.PI * 2;
      const d = size * 0.5;
      const v = 60 + 50 * Math.random();
      live.push({
        x: c + Math.cos(a) * d,
        y: c + Math.sin(a) * d,
        vx: -Math.cos(a) * v,
        vy: -Math.sin(a) * v,
        life: 1.2,
        color: byShare(p.intake.colors, p.intake.shares),
        size: 1.6,
      });
      if (i > 5) break;
    }
  }
  // New customers drift in from the edge, slower and bigger than data.
  if (p.arrivals) {
    const n = (0.15 + p.arrivals.rate) * 18 * dt;
    for (let i = 0; i < n || Math.random() < n - i; i++) {
      const a = Math.random() * Math.PI * 2;
      const d = size * 0.5;
      const v = 28 + 26 * Math.random();
      live.push({
        x: c + Math.cos(a) * d,
        y: c + Math.sin(a) * d,
        vx: -Math.cos(a) * v,
        vy: -Math.sin(a) * v,
        life: 2.2,
        color: byShare(p.arrivals.colors, p.arrivals.shares),
        size: 2.4,
      });
      if (i > 3) break;
    }
  }
  return live.slice(-MAX_PARTICLES);
}

/**
 * The AI as a living orb. It animates with requestAnimationFrame; under reduced
 * motion it draws one still frame, and without a canvas (tests) it draws nothing.
 */
export function Orb({
  input,
  size = 240,
  label,
}: {
  input: OrbInput;
  size?: number;
  label?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const key = JSON.stringify(input);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the input is compared by value (key)
  const params = useMemo(() => orbParams(input), [key]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas ? canDraw(canvas) : null;
    if (!canvas || !ctx) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = size * ratio;
    canvas.height = size * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (reducedMotion()) {
      drawFrame(ctx, size, params, 0, [], 1);
      return;
    }
    let particles: Particle[] = [];
    let frame = 0;
    let last = performance.now();
    const start = last;
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      particles = stepParticles(particles, params, size, dt);
      // An AI that often slips flickers now and then.
      const dim = Math.random() < params.flicker * 0.03 ? 0.55 : 1;
      drawFrame(ctx, size, params, (now - start) / 1000, particles, dim);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [params, size]);

  return (
    <canvas
      ref={ref}
      width={size}
      height={size}
      role="img"
      aria-label={label ?? "Your AI"}
      style={{ width: size, height: size }}
    />
  );
}
