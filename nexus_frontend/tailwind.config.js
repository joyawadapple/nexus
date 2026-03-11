/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        nexa: { blue: "#2563EB", dark: "#1E293B", slate: "#334155", light: "#F1F5F9" },
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
}

