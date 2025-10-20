/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./backend/app/templates/**/*.html",
    "./static/**/*.js",
    "./backend/app/static/**/*.js"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          900: '#1e3a8a'
        }
      }
    }
  },
  plugins: [],
}
