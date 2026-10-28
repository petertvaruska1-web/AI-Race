import { cleanup } from "@testing-library/react";
import { MotionGlobalConfig } from "motion/react";
import { afterEach } from "vitest";

// Animations finish at once in tests, so windows open and close without waiting.
MotionGlobalConfig.skipAnimations = true;

afterEach(() => cleanup());
