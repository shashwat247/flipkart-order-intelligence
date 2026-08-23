import type { Config } from "tailwindcss";

// "Night warehouse" design system — a sorting facility at 2am, not a generic
// dark SaaS dashboard. Every screen must consume these tokens; no off-token
// hex values or ad hoc spacing.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0A101C", // app background
          800: "#101827", // panel surface
          700: "#1A2438", // raised surface, table stripes
        },
        line: "#26324A", // hairline borders, grid lines
        slate: {
          400: "#8A99B5", // secondary text, axis labels
        },
        paper: "#E9EEF7", // primary text
        signal: "#2E7CF6", // primary action, links, active nav — exactly one per screen
        tape: "#FFB43D", // signature accent: graph rail, Medium risk, in-flight state
        flag: "#FF5A6E", // High risk, refusals, blocked injections
        verdant: "#31CFA3", // Low risk, grounded/passed checks
      },
      fontFamily: {
        display: ["Bricolage Grotesque", "system-ui", "sans-serif"],
        body: ["Public Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      spacing: {
        px8: "8px",
      },
      borderRadius: {
        control: "4px",
        panel: "8px",
        table: "0px",
      },
      transitionDuration: {
        node: "150ms",
      },
      transitionTimingFunction: {
        node: "cubic-bezier(0, 0, 0.2, 1)", // ease-out
      },
      keyframes: {
        // Chat messages settle in rather than popping — 150ms, same easing as
        // every other transition in the console.
        rise: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        dot: {
          "0%, 80%, 100%": { opacity: "0.25", transform: "translateY(0)" },
          "40%": { opacity: "1", transform: "translateY(-3px)" },
        },
      },
      animation: {
        rise: "rise 150ms cubic-bezier(0, 0, 0.2, 1) both",
        dot: "dot 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
