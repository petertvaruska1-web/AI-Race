import { render } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GlassBackground } from "../src/glass/GlassBackground.tsx";

/**
 * A stand-in WebGL context for the glass: it compiles and links until it is lost (as a real
 * context does after `loseContext()`), so a glass that loses its own context in a cleanup that
 * React runs twice (StrictMode) fails to start again.
 */
function fakeWebgl() {
  let lost = false;
  const ok = () => !lost;
  const gl = {
    VERTEX_SHADER: 1,
    FRAGMENT_SHADER: 2,
    COMPILE_STATUS: 3,
    LINK_STATUS: 4,
    ARRAY_BUFFER: 5,
    STATIC_DRAW: 6,
    FLOAT: 7,
    TRIANGLES: 8,
    createShader: () => ({}),
    shaderSource: () => {},
    compileShader: () => {},
    getShaderParameter: ok,
    createProgram: () => ({}),
    attachShader: () => {},
    linkProgram: () => {},
    getProgramParameter: ok,
    useProgram: () => {},
    createBuffer: () => ({}),
    bindBuffer: () => {},
    bufferData: () => {},
    getAttribLocation: () => 0,
    enableVertexAttribArray: () => {},
    vertexAttribPointer: () => {},
    getUniformLocation: () => ({}),
    viewport: () => {},
    uniform1f: () => {},
    uniform2f: () => {},
    uniform3fv: () => {},
    drawArrays: () => {},
    isContextLost: () => lost,
    getExtension: (name: string) =>
      name === "WEBGL_lose_context"
        ? {
            loseContext: () => {
              lost = true;
            },
          }
        : null,
  };
  return gl;
}

afterEach(() => vi.restoreAllMocks());

describe("the glass under React's StrictMode", () => {
  it("keeps drawing with WebGL after React mounts it twice", () => {
    const contexts = new WeakMap<HTMLCanvasElement, ReturnType<typeof fakeWebgl>>();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (
      this: HTMLCanvasElement,
    ) {
      const known = contexts.get(this);
      if (known) return known as never;
      const gl = fakeWebgl();
      contexts.set(this, gl);
      return gl as never;
    });
    const { container } = render(
      <StrictMode>
        <GlassBackground />
      </StrictMode>,
    );
    expect(container.querySelector("[data-glass-fallback]")).toBeNull();
    expect(container.querySelector("canvas")).not.toBeNull();
  });
});
