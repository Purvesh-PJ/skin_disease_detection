// Design tokens - Medical Green & Clean White (Light) / Dark Gray & Medical Green (Dark)

const colors = {
  // Medical Green & Emerald
  primary: {
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
  // Neutral slate & Dark Gray
  neutral: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
    950: '#0b0f19',
  },
  // Accent success
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
  // Accent error
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
  // Accent warning
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
  // Info colors (Medical Green Tint)
  info: {
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

// Light Theme (Clean White Paper & Medical Green)
export const lightTheme = {
  mode: 'light',
  colors: {
    ...colors,
    background: {
      primary: '#ffffff',
      secondary: '#f9fafb',
      tertiary: '#f3f4f6',
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
      light: '#f3f4f6',
      default: '#e5e7eb',
      dark: '#d1d5db',
      brand: '#bbf7d0',
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
      hover: '#f3f4f6',
      active: '#e5e7eb',
      selected: '#f0fdf4',
      selectedHover: '#dcfce7',
    },
    button: {
      primary: {
        bg: '#111827',
        bgHover: '#1f2937',
        bgActive: '#030712',
        text: '#ffffff',
      },
      secondary: {
        bg: '#f3f4f6',
        bgHover: '#e5e7eb',
        bgActive: '#d1d5db',
        text: '#111827',
        border: '#e5e7eb',
      },
      brand: {
        bg: '#16a34a',
        bgHover: '#15803d',
        bgActive: '#166534',
        text: '#ffffff',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(22, 163, 74, 0.08) 0%, rgba(249, 250, 251, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(22, 163, 74, 0.08) 0%, rgba(34, 197, 94, 0.12) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #f0fdf4 0%, #f9fafb 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.98)',
    authCardBorder: 'rgba(229, 231, 235, 0.9)',
    brandIcon: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
    progressBar: 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark Theme (Dark Gray & Medical Green)
export const darkTheme = {
  mode: 'dark',
  colors: {
    ...colors,
    background: {
      primary: '#111827',
      secondary: '#0b0f19',
      tertiary: '#1f2937',
      paper: '#1f2937',
      card: '#1f2937',
      elevated: '#374151',
    },
    text: {
      primary: '#f9fafb',
      secondary: '#cbd5e1',
      tertiary: '#94a3b8',
      inverse: '#111827',
    },
    border: {
      light: '#1f2937',
      default: '#374151',
      dark: '#4b5563',
      brand: 'rgba(34, 197, 94, 0.35)',
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
        bg: 'rgba(34, 197, 94, 0.15)',
        border: 'rgba(34, 197, 94, 0.3)',
        text: colors.primary[300],
        icon: colors.primary[400],
      },
    },
    interactive: {
      hover: '#1f2937',
      active: '#374151',
      selected: 'rgba(34, 197, 94, 0.15)',
      selectedHover: 'rgba(34, 197, 94, 0.25)',
    },
    button: {
      primary: {
        bg: '#f9fafb',
        bgHover: '#e5e7eb',
        bgActive: '#d1d5db',
        text: '#111827',
      },
      secondary: {
        bg: '#1f2937',
        bgHover: '#374151',
        bgActive: '#4b5563',
        text: '#f9fafb',
        border: '#374151',
      },
      brand: {
        bg: '#16a34a',
        bgHover: '#22c55e',
        bgActive: '#15803d',
        text: '#ffffff',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(34, 197, 94, 0.15) 0%, rgba(11, 15, 25, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.15) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #111827 0%, #0b0f19 100%)',
    authCardBg: 'rgba(31, 41, 55, 0.95)',
    authCardBorder: 'rgba(255, 255, 255, 0.1)',
    brandIcon: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
    progressBar: 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

const theme = { lightTheme, darkTheme };
export default theme;
