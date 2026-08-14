// Design tokens - Clean, Professional Academic / Engineering Documentation Aesthetic

const colors = {
  // Calm Engineering Blue
  primary: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a',
  },
  // Neutral slate (Ultra-clean, readable paper grays)
  neutral: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    300: '#cbd5e1',
    400: '#94a3b8',
    500: '#64748b',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
    950: '#020617',
  },
  // Accent success (Clean emerald)
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',
  },
  // Accent error (Crimson)
  error: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d',
  },
  // Accent warning (Amber)
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
  },
  // Info colors
  info: {
    50: '#eff6ff',
    100: '#dbeafe',
    200: '#bfdbfe',
    300: '#93c5fd',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8',
    800: '#1e40af',
    900: '#1e3a8a',
  },
};

// Spacing scale (in pixels)
const spacing = {
  0: '0',
  1: '4px',
  1.5: '6px',
  2: '8px',
  2.5: '10px',
  3: '12px',
  3.5: '14px',
  4: '16px',
  4.5: '18px',
  5: '20px',
  6: '24px',
  7: '28px',
  8: '32px',
  10: '40px',
  12: '48px',
  16: '64px',
  20: '80px',
  24: '96px',
  32: '128px',
};

// Border radius scale
const borderRadius = {
  none: '0',
  sm: '6px',
  md: '10px',
  lg: '14px',
  xl: '18px',
  '2xl': '22px',
  '3xl': '28px',
  card: '20px',
  container: '24px',
  pill: '9999px',
  full: '9999px',
};

// Clean Elevation Shadows
const shadows = {
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.04)',
  md: '0 4px 12px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02)',
  lg: '0 10px 24px rgba(0, 0, 0, 0.06), 0 2px 6px rgba(0, 0, 0, 0.02)',
  xl: '0 20px 36px rgba(0, 0, 0, 0.08)',
  paper: '0 1px 3px rgba(0, 0, 0, 0.04), 0 6px 18px rgba(0, 0, 0, 0.03)',
  hover: '0 12px 28px rgba(0, 0, 0, 0.08)',
  card: '0 2px 12px rgba(0, 0, 0, 0.04)',
  floating: '0 18px 40px -8px rgba(0, 0, 0, 0.1), 0 0 1px 1px rgba(0, 0, 0, 0.04)',
  subtle: 'rgba(0, 0, 0, 0.03) 0px 2px 12px 0px, rgba(0, 0, 0, 0.06) 0px 0px 0px 1px',
};

// Breakpoints
const breakpoints = {
  xs: '480px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
};

// Transitions
const transitions = {
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
};

// Light Theme (Clean White Paper)
export const lightTheme = {
  mode: 'light',
  colors: {
    ...colors,
    background: {
      primary: '#ffffff',
      secondary: '#f8fafc',
      tertiary: '#f1f5f9',
      paper: '#ffffff',
      card: '#ffffff',
      elevated: '#ffffff',
    },
    text: {
      primary: '#0f172a',
      secondary: '#475569',
      tertiary: '#94a3b8',
      inverse: '#ffffff',
    },
    border: {
      light: '#f1f5f9',
      default: '#e2e8f0',
      dark: '#cbd5e1',
      brand: '#bfdbfe',
    },
    status: {
      success: {
        bg: colors.success[50],
        border: colors.success[200],
        text: colors.success[700],
        icon: colors.success[600],
      },
      error: {
        bg: colors.error[50],
        border: colors.error[200],
        text: colors.error[700],
        icon: colors.error[600],
      },
      warning: {
        bg: colors.warning[50],
        border: colors.warning[200],
        text: colors.warning[800],
        icon: colors.warning[600],
      },
      info: {
        bg: colors.info[50],
        border: colors.info[200],
        text: colors.info[700],
        icon: colors.info[600],
      },
    },
    interactive: {
      hover: '#f1f5f9',
      active: '#e2e8f0',
      selected: '#eff6ff',
      selectedHover: '#dbeafe',
    },
    button: {
      primary: {
        bg: '#0f172a',
        bgHover: '#1e293b',
        bgActive: '#020617',
        text: '#ffffff',
      },
      secondary: {
        bg: '#f1f5f9',
        bgHover: '#e2e8f0',
        bgActive: '#cbd5e1',
        text: '#0f172a',
        border: '#e2e8f0',
      },
      brand: {
        bg: '#2563eb',
        bgHover: '#1d4ed8',
        bgActive: '#1e40af',
        text: '#ffffff',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.08) 0%, rgba(248, 250, 252, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #eff6ff 0%, #f8fafc 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.98)',
    authCardBorder: 'rgba(226, 232, 240, 0.9)',
    brandIcon: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
    progressBar: 'linear-gradient(90deg, #2563eb 0%, #3b82f6 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark Theme (Clean Slate Obsidian)
export const darkTheme = {
  mode: 'dark',
  colors: {
    ...colors,
    background: {
      primary: '#0f172a',
      secondary: '#090d16',
      tertiary: '#1e293b',
      paper: '#1e293b',
      card: '#1e293b',
      elevated: '#334155',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5e1',
      tertiary: '#94a3b8',
      inverse: '#0f172a',
    },
    border: {
      light: '#1e293b',
      default: '#334155',
      dark: '#475569',
      brand: 'rgba(37, 99, 235, 0.4)',
    },
    status: {
      success: {
        bg: 'rgba(34, 197, 94, 0.15)',
        border: 'rgba(34, 197, 94, 0.3)',
        text: colors.success[400],
        icon: colors.success[400],
      },
      error: {
        bg: 'rgba(239, 68, 68, 0.15)',
        border: 'rgba(239, 68, 68, 0.3)',
        text: colors.error[400],
        icon: colors.error[400],
      },
      warning: {
        bg: 'rgba(245, 158, 11, 0.15)',
        border: 'rgba(245, 158, 11, 0.3)',
        text: colors.warning[400],
        icon: colors.warning[400],
      },
      info: {
        bg: 'rgba(37, 99, 235, 0.15)',
        border: 'rgba(37, 99, 235, 0.3)',
        text: colors.primary[300],
        icon: colors.primary[400],
      },
    },
    interactive: {
      hover: '#1e293b',
      active: '#334155',
      selected: 'rgba(37, 99, 235, 0.15)',
      selectedHover: 'rgba(37, 99, 235, 0.25)',
    },
    button: {
      primary: {
        bg: '#f8fafc',
        bgHover: '#e2e8f0',
        bgActive: '#cbd5e1',
        text: '#0f172a',
      },
      secondary: {
        bg: '#1e293b',
        bgHover: '#334155',
        bgActive: '#475569',
        text: '#f8fafc',
        border: '#334155',
      },
      brand: {
        bg: '#2563eb',
        bgHover: '#3b82f6',
        bgActive: '#1d4ed8',
        text: '#ffffff',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.15) 0%, rgba(9, 13, 22, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%)',
    authCardBg: 'rgba(30, 41, 59, 0.95)',
    authCardBorder: 'rgba(255, 255, 255, 0.1)',
    brandIcon: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
    progressBar: 'linear-gradient(90deg, #2563eb 0%, #3b82f6 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

const theme = { lightTheme, darkTheme };
export default theme;
