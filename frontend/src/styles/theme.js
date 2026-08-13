// Theme configuration for the application
const colors = {
  // Primary colors - Tailwind Sky
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
  // Neutral/Gray colors
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
  },
  // Success colors
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
  // Error colors
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
};

// Border radius
const borderRadius = {
  none: '0',
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  '2xl': '20px',
  '3xl': '24px',
  full: '9999px',
};

// Shadows
const shadows = {
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  card: 'rgba(149, 157, 165, 0.2) 0px 8px 24px',
  subtle: 'rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px',
};

// Breakpoints
const breakpoints = {
  xs: '480px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
};

// Transitions
const transitions = {
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
};

// Light theme
export const lightTheme = {
  mode: 'light',
  colors: {
    ...colors,
    background: {
      primary: '#fafbff',
      secondary: '#f0f4ff',
      tertiary: '#e8eeff',
    },
    text: {
      primary: '#1a1a2e',
      secondary: '#4a4a68',
      tertiary: '#7a7a98',
      inverse: '#ffffff',
    },
    border: {
      light: '#e0e5f0',
      default: '#c8d0e8',
      dark: '#a0a8c8',
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
      hover: colors.neutral[100],
      active: colors.neutral[200],
      selected: colors.primary[50],
      selectedHover: colors.primary[100],
    },
    button: {
      primary: {
        bg: colors.primary[600],
        bgHover: colors.primary[700],
        bgActive: colors.primary[800],
        text: '#ffffff',
      },
      secondary: {
        bg: colors.neutral[100],
        bgHover: colors.neutral[200],
        bgActive: colors.neutral[300],
        text: colors.neutral[800],
        border: colors.neutral[300],
      },
    },
  },
  gradients: {
    authBg: 'radial-gradient(circle at 50% 0%, #e0f2fe 0%, #fafbff 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.85)',
    authCardBorder: 'rgba(226, 232, 240, 0.8)',
    brandIcon: 'linear-gradient(135deg, #0ea5e9, #0369a1)',
    progressBar: 'linear-gradient(90deg, #0ea5e9, #0284c7)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark theme
export const darkTheme = {
  mode: 'dark',
  colors: {
    ...colors,
    background: {
      primary: '#1a1a2e',
      secondary: '#16162a',
      tertiary: '#252542',
    },
    text: {
      primary: '#f1f5f9',
      secondary: '#cbd5e1',
      tertiary: '#94a3b8',
      inverse: '#0f172a',
    },
    border: {
      light: '#2d2d4a',
      default: '#3d3d5c',
      dark: '#4d4d6a',
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
      hover: '#252542',
      active: '#2d2d4a',
      selected: 'rgba(59, 130, 246, 0.15)',
      selectedHover: 'rgba(59, 130, 246, 0.25)',
    },
    button: {
      primary: {
        bg: colors.primary[500],
        bgHover: colors.primary[400],
        bgActive: colors.primary[600],
        text: '#ffffff',
      },
      secondary: {
        bg: '#2d2d4a',
        bgHover: '#3d3d5c',
        bgActive: '#4d4d6a',
        text: '#f1f5f9',
        border: '#3d3d5c',
      },
    },
  },
  gradients: {
    authBg: 'radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%)',
    authCardBg: 'rgba(30, 41, 59, 0.75)',
    authCardBorder: 'rgba(255, 255, 255, 0.1)',
    brandIcon: 'linear-gradient(135deg, #0ea5e9, #0369a1)',
    progressBar: 'linear-gradient(90deg, #0ea5e9, #0284c7)',
  },
  spacing,
  borderRadius,
  shadows: {
    ...shadows,
    card: 'rgba(0, 0, 0, 0.3) 0px 8px 24px',
    subtle: 'rgba(0, 0, 0, 0.2) 0px 6px 24px 0px, rgba(255, 255, 255, 0.05) 0px 0px 0px 1px',
  },
  breakpoints,
  transitions,
};

export default lightTheme;
