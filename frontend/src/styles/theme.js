// Design tokens - Modern "White Paper" Aesthetic (Apple / Android Developers Inspired)

const colors = {
  // Primary brand (Deep Medical Azure)
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
  // Neutral slate (Ultra-clean, crisp paper grays)
  neutral: {
    50: '#fbfbfd',
    100: '#f4f4f7',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
  // Accent colors (Emerald success)
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
  // Error colors (Crimson coral)
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
  // Warning colors
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
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
};

// Spacing scale (in pixels)
const spacing = {
  0: '0',
  1: '4px',
  2: '8px',
  3: '12px',
  4: '16px',
  5: '20px',
  6: '24px',
  8: '32px',
  10: '40px',
  12: '48px',
  16: '64px',
  20: '80px',
  24: '96px',
  32: '128px',
};

// Hyper-rounded border radius scale
const borderRadius = {
  none: '0',
  sm: '6px',
  md: '10px',
  lg: '14px',
  xl: '18px',
  '2xl': '24px',
  '3xl': '32px',
  card: '24px',
  container: '28px',
  pill: '9999px',
  full: '9999px',
};

// Apple / Material You Elevation Shadows
const shadows = {
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
  md: '0 4px 12px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)',
  lg: '0 8px 24px rgba(0, 0, 0, 0.06), 0 2px 6px rgba(0, 0, 0, 0.03)',
  xl: '0 16px 36px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.03)',
  paper: '0 1px 3px rgba(0, 0, 0, 0.05), 0 10px 30px -5px rgba(0, 0, 0, 0.04)',
  hover: '0 14px 34px -4px rgba(0, 0, 0, 0.1), 0 4px 12px -2px rgba(0, 0, 0, 0.04)',
  card: '0 2px 16px -2px rgba(0, 0, 0, 0.05)',
  floating: '0 20px 48px -10px rgba(0, 0, 0, 0.12), 0 0 1px 1px rgba(0, 0, 0, 0.04)',
  subtle: 'rgba(0, 0, 0, 0.04) 0px 4px 20px 0px, rgba(0, 0, 0, 0.06) 0px 0px 0px 1px',
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
  fast: '150ms cubic-bezier(0.16, 1, 0.3, 1)',
  normal: '220ms cubic-bezier(0.16, 1, 0.3, 1)',
  slow: '350ms cubic-bezier(0.16, 1, 0.3, 1)',
};

// Light Theme (Clean White Paper Aesthetic)
export const lightTheme = {
  mode: 'light',
  colors: {
    ...colors,
    background: {
      primary: '#ffffff',
      secondary: '#fbfbfd',
      tertiary: '#f4f4f7',
      paper: '#ffffff',
      card: '#ffffff',
      elevated: '#ffffff',
    },
    text: {
      primary: '#111827',
      secondary: '#4b5563',
      tertiary: '#9ca3af',
      inverse: '#ffffff',
    },
    border: {
      light: '#f1f3f5',
      default: '#e5e7eb',
      dark: '#d1d5db',
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
      hover: '#f4f4f7',
      active: '#e5e7eb',
      selected: '#f0f9ff',
      selectedHover: '#e0f2fe',
    },
    button: {
      primary: {
        bg: '#111827',
        bgHover: '#1f2937',
        bgActive: '#030712',
        text: '#ffffff',
      },
      secondary: {
        bg: '#f4f4f7',
        bgHover: '#e5e7eb',
        bgActive: '#d1d5db',
        text: '#111827',
        border: '#e5e7eb',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(14, 165, 233, 0.12) 0%, rgba(255, 255, 255, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(59, 130, 246, 0.12) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #f0f9ff 0%, #fbfbfd 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.95)',
    authCardBorder: 'rgba(229, 231, 235, 0.8)',
    brandIcon: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
    progressBar: 'linear-gradient(90deg, #0ea5e9, #0284c7)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark Theme (Obsidian White-Paper Inverted)
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
        bg: 'rgba(14, 165, 233, 0.15)',
        border: 'rgba(14, 165, 233, 0.3)',
        text: colors.info[400],
        icon: colors.info[400],
      },
    },
    interactive: {
      hover: '#1e293b',
      active: '#334155',
      selected: 'rgba(14, 165, 233, 0.15)',
      selectedHover: 'rgba(14, 165, 233, 0.25)',
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
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(14, 165, 233, 0.18) 0%, rgba(15, 23, 42, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(59, 130, 246, 0.2) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%)',
    authCardBg: 'rgba(30, 41, 59, 0.85)',
    authCardBorder: 'rgba(255, 255, 255, 0.1)',
    brandIcon: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
    progressBar: 'linear-gradient(90deg, #0ea5e9, #0284c7)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

const theme = { lightTheme, darkTheme };
export default theme;
