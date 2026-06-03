/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        obs: {
          bg: "#07111F",
          card: "#0E1B2E",
          card2: "#101F35",
          border: "#1F3554",
          text: "#EAF2FF",
          muted: "#8FA6C2",
          positive: "#22C55E",
          negative: "#EF4444",
          warning: "#FACC15",
          blue: "#38BDF8",
          violet: "#8B5CF6"
        }
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"]
      }
    }
  },
  plugins: []
};

