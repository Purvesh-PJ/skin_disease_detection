// Design tokens - Android Developers & Material 3 Expressive Aesthetic

const colors = {
  // Android Green & Emerald (Vibrant signature accent)
  emerald: {
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
    android: '#3ddc84', // Iconic Android Green
    androidDark: '#00875a',
  },
  // Primary brand (Deep Medical Azure & Indigo)
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
  // Deep Pine & Slate (Signature Android Developers Dark surfaces)
  pine: {
    900: '#073042',
    800: '#0a3d54',
    700: '#0e4e6c',
    600: '#136287',
  },
  // Indigo / Periwinkle (Material 3 Expressive)
  indigo: {
    50: '#eef2ff',
    100: '#e0e7ff',
    200: '#c7d2fe',
    300: '#a5b4fc',
    400: '#818cf8',
    500: '#6366f1',
    600: '#4f46e5',
    700: '#4338ca',
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
    950: '#0b0f19',
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
  // Warning colors (Warm Amber)
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

// Hyper-rounded squircle border radius scale
const borderRadius = {
  none: '0',
  sm: '8px',
  md: '12px',
  lg: '16px',
  xl: '20px',
  '2xl': '24px',
  '3xl': '32px',
  card: '24px',
  container: '32px',
  bento: '32px',
  pill: '9999px',
  full: '9999px',
};

// Material 3 & Android Developers Elevation Shadows
const shadows = {
  none: 'none',
  sm: '0 1px 3px rgba(0, 0, 0, 0.04)',
  md: '0 4px 14px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.02)',
  lg: '0 10px 28px rgba(0, 0, 0, 0.06), 0 2px 8px rgba(0, 0, 0, 0.03)',
  xl: '0 20px 40px rgba(0, 0, 0, 0.08), 0 4px 14px rgba(0, 0, 0, 0.03)',
  bento: '0 2px 20px -2px rgba(0, 0, 0, 0.04), 0 0 1px 1px rgba(0, 0, 0, 0.03)',
  paper: '0 1px 3px rgba(0, 0, 0, 0.04), 0 12px 32px -4px rgba(0, 0, 0, 0.05)',
  hover: '0 16px 40px -4px rgba(0, 0, 0, 0.1), 0 6px 16px -2px rgba(0, 0, 0, 0.05)',
  card: '0 2px 16px -2px rgba(0, 0, 0, 0.05)',
  floating: '0 24px 54px -12px rgba(0, 0, 0, 0.14), 0 0 1px 1px rgba(0, 0, 0, 0.04)',
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
  fast: '150ms cubic-bezier(0.2, 0, 0, 1)',
  normal: '240ms cubic-bezier(0.2, 0, 0, 1)',
  slow: '380ms cubic-bezier(0.2, 0, 0, 1)',
};

// Light Theme (Clean White Paper + Material 3 Pastel Tonal Surfaces)
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
      tonalMint: '#f0fdf4',
      tonalIndigo: '#eef2ff',
      tonalSand: '#fffbeb',
      tonalIce: '#f0f9ff',
      tonalPurple: '#faf5ff',
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
      hover: '#f4f4f7',
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
        bg: '#f4f4f7',
        bgHover: '#e5e7eb',
        bgActive: '#d1d5db',
        text: '#111827',
        border: '#e5e7eb',
      },
      android: {
        bg: '#3ddc84',
        bgHover: '#22c55e',
        bgActive: '#16a34a',
        text: '#073042',
      },
      pine: {
        bg: '#073042',
        bgHover: '#0a3d54',
        bgActive: '#052230',
        text: '#ffffff',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(61, 220, 132, 0.15) 0%, rgba(255, 255, 255, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(61, 220, 132, 0.12) 0%, rgba(14, 165, 233, 0.12) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #f0fdf4 0%, #fbfbfd 100%)',
    authCardBg: 'rgba(255, 255, 255, 0.95)',
    authCardBorder: 'rgba(229, 231, 235, 0.8)',
    brandIcon: 'linear-gradient(135deg, #073042 0%, #0a3d54 100%)',
    progressBar: 'linear-gradient(90deg, #3ddc84 0%, #059669 100%)',
    bentoPine: 'linear-gradient(135deg, #073042 0%, #0b0f19 100%)',
    bentoMint: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
    bentoIndigo: 'linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

// Dark Theme (Android Developers Obsidian Pine)
export const darkTheme = {
  mode: 'dark',
  colors: {
    ...colors,
    background: {
      primary: '#0b0f19',
      secondary: '#070a10',
      tertiary: '#131b2e',
      paper: '#131b2e',
      card: '#131b2e',
      elevated: '#1e293b',
      tonalMint: 'rgba(34, 197, 94, 0.08)',
      tonalIndigo: 'rgba(99, 102, 241, 0.08)',
      tonalSand: 'rgba(245, 158, 11, 0.08)',
      tonalIce: 'rgba(14, 165, 233, 0.08)',
      tonalPurple: 'rgba(168, 85, 247, 0.08)',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5e1',
      tertiary: '#94a3b8',
      inverse: '#0b0f19',
    },
    border: {
      light: '#1e293b',
      default: '#334155',
      dark: '#475569',
      brand: 'rgba(61, 220, 132, 0.3)',
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
      hover: '#131b2e',
      active: '#1e293b',
      selected: 'rgba(61, 220, 132, 0.15)',
      selectedHover: 'rgba(61, 220, 132, 0.25)',
    },
    button: {
      primary: {
        bg: '#f8fafc',
        bgHover: '#e2e8f0',
        bgActive: '#cbd5e1',
        text: '#0b0f19',
      },
      secondary: {
        bg: '#131b2e',
        bgHover: '#1e293b',
        bgActive: '#334155',
        text: '#f8fafc',
        border: '#334155',
      },
      android: {
        bg: '#3ddc84',
        bgHover: '#4ade80',
        bgActive: '#22c55e',
        text: '#073042',
      },
      pine: {
        bg: '#073042',
        bgHover: '#0a3d54',
        bgActive: '#0e4e6c',
        text: '#3ddc84',
      },
    },
  },
  gradients: {
    heroGlow: 'radial-gradient(circle at 50% 0%, rgba(61, 220, 132, 0.18) 0%, rgba(11, 15, 25, 0) 70%)',
    heroBadge: 'linear-gradient(135deg, rgba(61, 220, 132, 0.15) 0%, rgba(14, 165, 233, 0.15) 100%)',
    authBg: 'radial-gradient(circle at 50% 0%, #073042 0%, #0b0f19 100%)',
    authCardBg: 'rgba(19, 27, 46, 0.9)',
    authCardBorder: 'rgba(255, 255, 255, 0.1)',
    brandIcon: 'linear-gradient(135deg, #073042 0%, #0a3d54 100%)',
    progressBar: 'linear-gradient(90deg, #3ddc84 0%, #059669 100%)',
    bentoPine: 'linear-gradient(135deg, #073042 0%, #051a24 100%)',
    bentoMint: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%)',
    bentoIndigo: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%)',
  },
  spacing,
  borderRadius,
  shadows,
  breakpoints,
  transitions,
};

const theme = { lightTheme, darkTheme };
export default theme;
