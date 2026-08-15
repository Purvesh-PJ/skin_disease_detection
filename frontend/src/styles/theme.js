// Design tokens - Medical Green & Clean White (Light) / True Dark Gray & Medical Green (Dark)

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
  // Neutral Clean Slate (Light Mode) & Pure Dark Gray (Dark Mode)
  neutral: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#e5e5e5',
    300: '#d4d4d4',
    400: '#a3a3a3',
    500: '#737373',
    600: '#525252',
    700: '#404040',
    800: '#262626',
    900: '#171717',
    950: '#0a0a0a',
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

// Font families (Poppins primary)
const fontFamily = {
  heading: '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  body: '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
};

// Light Theme (Clean White Paper & Medical Green)
export const lightTheme = {
  mode: 'light',
  fontFamily,
  colors: {
    ...colors,
    background: {
      primary: '#ffffff',
      secondary: '#fafafa',
      tertiary: '#f5f5f5',
      paper: '#ffffff',
      card: '#ffffff',
      elevated: '#ffffff',
    },
    text: {
      primary: '#171717',
      secondary: '#525252',
      tertiary: '#a3a3a3',
      inverse: '#ffffff',
    },
    border: {
      light: '#f5f5f5',
      default: '#e5e5e5',
      dark: '#d4d4d4',
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
      hover: '#f5f5f5',
      active: '#e5e5e5',
      selected: '#f0fdf4',
      selectedHover: '#dcfce7',
    },
    button: {
      primary: {
        bg: '#171717',
        bgHover: '#262626',
        bgActive: '#0a0a0a',
        text: '#ffffff',
      },
      secondary: {
        bg: '#f5f5f5',
        bgHover: '#e5e5e5',
        bgActive: '#d4d4d4',
        text: '#171717',
        border: '#e5e5e5',
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
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(22, 163, 74, 0.08) 0%, rgba(250, 250, 250, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(22, 163, 74, 0.08) 0%, rgba(34, 197, 94, 0.12) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #f0fdf4 0%, #fafafa 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.98)',
    authCardBorder: 'rgba(229, 229, 229, 0.9)',
    brandIcon: 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
    progressBar: 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark Theme (True Dark Gray / Charcoal - Zero Blue Tint)
export const darkTheme = {
  mode: 'dark',
  fontFamily,
  colors: {
    ...colors,
    background: {
      primary: '#121212',
      secondary: '#0a0a0a',
      tertiary: '#1c1c1c',
      paper: '#181818',
      card: '#181818',
      elevated: '#242424',
    },
    text: {
      primary: '#f5f5f5',
      secondary: '#a3a3a3',
      tertiary: '#737373',
      inverse: '#0a0a0a',
    },
    border: {
      light: '#1f1f1f',
      default: '#2a2a2a',
      dark: '#3a3a3a',
      brand: 'rgba(34, 197, 94, 0.4)',
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
      hover: '#1e1e1e',
      active: '#282828',
      selected: 'rgba(34, 197, 94, 0.15)',
      selectedHover: 'rgba(34, 197, 94, 0.25)',
    },
    button: {
      primary: {
        bg: '#f5f5f5',
        bgHover: '#e5e5e5',
        bgActive: '#d4d4d4',
        text: '#0a0a0a',
      },
      secondary: {
        bg: '#1e1e1e',
        bgHover: '#282828',
        bgActive: '#333333',
        text: '#f5f5f5',
        border: '#2a2a2a',
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
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(34, 197, 94, 0.12) 0%, rgba(10, 10, 10, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(22, 163, 74, 0.12) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #1a1a1a 0%, #0a0a0a 100%)',
    authCardBg: 'rgba(24, 24, 24, 0.96)',
    authCardBorder: 'rgba(255, 255, 255, 0.08)',
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
