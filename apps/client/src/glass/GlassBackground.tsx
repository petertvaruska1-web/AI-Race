import { useEffect, useRef, useState } from "react";
import { createFrameMeter, due, framePolicy } from "../graphics/frames.ts";
import { reducedMotion } from "../motion.ts";
import { useGraphics } from "../store/graphics.ts";
import { glassFor, useLook } from "../theme/glass.ts";
import { NEUTRAL_PALETTE } from "../theme/palette.ts";
import { webglAvailable } from "../three/support.ts";
import { GLASS_FRAGMENT, GLASS_VERTEX } from "./shader.ts";

/** The glass tiles' size in CSS pixels (design §6). */
const CELL_PX = 15;

const rgb = (hex: string) => {
  const n = Number.parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255] as const;
};

type Painter = {
  draw(o: { time: number; light: number; mouse: [number, number]; power: number }): void;
  resize(): void;
  setColors(primary: string, secondary: string): void;
  dispose(): void;
};

/** A WebGL painter for the glass, or null when it can't start. */
function painter(canvas: HTMLCanvasElement): Painter | null {
  const gl = canvas.getContext("webgl", { antialias: false, alpha: false });
  if (!gl) return null;
  const shader = (type: number, src: string) => {
    const s = gl.createShader(type);
    if (!s) return null;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  };
  const vs = shader(gl.VERTEX_SHADER, GLASS_VERTEX);
  const fs = shader(gl.FRAGMENT_SHADER, GLASS_FRAGMENT);
  const program = gl.createProgram();
  if (!vs || !fs || !program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
  // biome-ignore lint/correctness/useHookAtTopLevel: WebGL's useProgram, not a React hook
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const at = gl.getAttribLocation(program, "a");
  gl.enableVertexAttribArray(at);
  gl.vertexAttribPointer(at, 2, gl.FLOAT, false, 0, 0);
  const u = (name: string) => gl.getUniformLocation(program, name);
  const U = {
    res: u("uRes"),
    time: u("uTime"),
    p: u("uP"),
    s: u("uS"),
    light: u("uLight"),
    cell: u("uCell"),
    mouse: u("uMouse"),
    power: u("uPower"),
  };
  // Half the device's pixels: the tiles hide it, and the card stays free for training.
  let scale = 0.5;
  return {
    resize() {
      scale = Math.max(0.5, (window.devicePixelRatio || 1) * 0.5);
      canvas.width = Math.max(1, Math.round(window.innerWidth * scale));
      canvas.height = Math.max(1, Math.round(window.innerHeight * scale));
      gl.viewport(0, 0, canvas.width, canvas.height);
    },
    setColors(primary, secondary) {
      gl.uniform3fv(U.p, rgb(primary));
      gl.uniform3fv(U.s, rgb(secondary));
    },
    draw({ time, light, mouse, power }) {
      gl.uniform2f(U.res, canvas.width, canvas.height);
      gl.uniform1f(U.time, time);
      gl.uniform1f(U.light, light);
      gl.uniform1f(U.cell, Math.max(4, Math.round(CELL_PX * scale)));
      gl.uniform2f(U.mouse, mouse[0], mouse[1]);
      gl.uniform1f(U.power, power);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() {
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}

/**
 * The pixelated lab glass with the aurora behind every screen (design §6, C57), in the company's
 * colors. Draws at most 30 frames a second, nothing while the tab is hidden, and one still frame
 * for reduced motion and lighter graphics; without WebGL, a CSS glass in the same colors.
 */
export function GlassBackground() {
  const palette = useLook((s) => s.palette) ?? NEUTRAL_PALETTE;
  const glass = glassFor(palette.primary);
  const lighter = useGraphics((s) => s.lighter);
  const [webgl, setWebgl] = useState(() => webglAvailable(1));
  const canvas = useRef<HTMLCanvasElement>(null);
  const look = useRef({ palette, light: glass === "light" ? 1 : 0, lighter });
  look.current = { palette, light: glass === "light" ? 1 : 0, lighter };
  const dirty = useRef(true);

  // biome-ignore lint/correctness/useExhaustiveDependencies: a new palette or setting redraws a still glass
  useEffect(() => {
    dirty.current = true;
  }, [palette.primary, palette.secondary, glass, lighter]);

  useEffect(() => {
    const el = canvas.current;
    if (!webgl || !el) return;
    const paint = painter(el);
    if (!paint) {
      setWebgl(false);
      return;
    }
    const meter = createFrameMeter();
    let light = look.current.light;
    let mouse: [number, number] = [0.5, 0.5];
    let target: [number, number] = [0.5, 0.5];
    let last = 0;
    let frame = 0;
    const start = performance.now();
    const onResize = () => {
      paint.resize();
      dirty.current = true;
    };
    const onPointer = (e: PointerEvent) => {
      target = [e.clientX / window.innerWidth, 1 - e.clientY / window.innerHeight];
    };
    onResize();
    window.addEventListener("resize", onResize);
    window.addEventListener("pointermove", onPointer);
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      const policy = framePolicy({
        reducedMotion: reducedMotion(),
        lighter: look.current.lighter,
        hidden: document.visibilityState === "hidden",
        interacting: false,
      });
      const { palette: p, light: goal } = look.current;
      if (policy.still) {
        if (!dirty.current || document.visibilityState === "hidden") return;
        light = goal;
      } else {
        if (!due(now, last, policy.fps)) return;
        if (last > 0 && meter.record(now - last)) useGraphics.getState().setLighter(true, true);
        light += (goal - light) * 0.08;
        mouse = [
          mouse[0] + (target[0] - mouse[0]) * 0.08,
          mouse[1] + (target[1] - mouse[1]) * 0.08,
        ];
      }
      dirty.current = false;
      last = now;
      paint.setColors(p.primary, p.secondary);
      paint.draw({ time: (now - start) / 1000, light, mouse, power: 1 });
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onPointer);
      paint.dispose();
    };
  }, [webgl]);

  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10">
      {webgl ? (
        <canvas ref={canvas} className="block h-full w-full" />
      ) : (
        <div data-glass-fallback data-glass={glass} className="glass-fallback h-full w-full" />
      )}
    </div>
  );
}
