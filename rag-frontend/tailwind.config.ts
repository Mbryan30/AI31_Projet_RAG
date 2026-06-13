import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['DM Sans', 'system-ui', 'sans-serif'],
        serif: ['DM Serif Display', 'Georgia', 'serif'],
        mono:  ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        bg: {
          DEFAULT: '#0a0a0b',
          1: '#111113',
          2: '#18181c',
          3: '#222228',
        },
        border: {
          DEFAULT: 'rgba(255,255,255,0.07)',
          md:      'rgba(255,255,255,0.12)',
        },
        accent: {
          DEFAULT: '#6c8fff',
          dark:    '#4a6ef0',
          glow:    'rgba(108,143,255,0.15)',
        },
        tx: {
          DEFAULT: '#e8e8ea',
          2: '#9090a0',
          3: '#505060',
        },
        success: '#3ddc84',
        warning: '#f5a623',
        danger:  '#ff5c5c',
      },
      borderRadius: {
        DEFAULT: '12px',
        sm: '8px',
        lg: '16px',
      },
      keyframes: {
        fadeUp: {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%,100%': { opacity: '0.3' },
          '50%':     { opacity: '1' },
        },
        slideIn: {
          from: { opacity: '0', transform: 'translateX(-8px)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
      },
      animation: {
        fadeUp:  'fadeUp 0.3s ease both',
        blink:   'blink 1.2s ease-in-out infinite',
        slideIn: 'slideIn 0.2s ease both',
      },
    },
  },
  plugins: [],
} satisfies Config
