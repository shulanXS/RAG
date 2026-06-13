import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: 'var(--color-primary)',
          hover: 'var(--color-primary-hover)',
        },
        background: 'var(--color-bg)',
        'background-secondary': 'var(--color-bg-secondary)',
        border: 'var(--color-border)',
        foreground: 'var(--color-text)',
        'foreground-secondary': 'var(--color-text-secondary)',
      },
    },
  },
  plugins: [],
}

export default config
