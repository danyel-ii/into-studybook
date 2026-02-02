import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        ink: "#0b0f19",
        glass: "rgba(255,255,255,0.08)",
        glassStrong: "rgba(255,255,255,0.14)",
        accent: "#3dd6a6",
        accentSoft: "#5ee4c2",
      },
      fontFamily: {
        sans: ["var(--font-sora)", "system-ui", "sans-serif"],
        display: ["var(--font-fraunces)", "serif"],
      },
      boxShadow: {
        glass: "0 10px 40px rgba(7, 12, 23, 0.25)",
      },
      backdropBlur: {
        glass: "18px",
      },
    },
  },
  plugins: [],
};

export default config;
