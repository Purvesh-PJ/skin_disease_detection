import styled, { css } from 'styled-components';

// Typography scale - Editorial White Paper
export const typography = {
  fontFamily: {
    heading: '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    body: '"Poppins", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
  fontSize: {
    xs: '0.75rem',     // 12px
    sm: '0.875rem',    // 14px
    base: '1rem',      // 16px
    lg: '1.125rem',    // 18px
    xl: '1.25rem',     // 20px
    '2xl': '1.5rem',   // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem',  // 36px
    '5xl': '3rem',     // 48px
    '6xl': '3.75rem',  // 60px
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
  },
  lineHeight: {
    tight: 1.15,
    snug: 1.25,
    normal: 1.5,
    relaxed: 1.65,
  },
  letterSpacing: {
    tighter: '-0.035em',
    tight: '-0.02em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
  },
};

// Base text styles mixin
const baseTextStyles = css`
  margin: 0;
  font-family: ${typography.fontFamily.body};
`;

// Heading styles mixin
const headingStyles = css`
  ${baseTextStyles}
  font-family: ${typography.fontFamily.heading};
  letter-spacing: -0.02em;
  color: ${({ theme, color }) => color || theme.colors.text.primary};
`;

export const DisplayTitle = styled.h1`
  ${headingStyles}
  font-size: ${typography.fontSize['5xl']};
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.035em;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    font-size: ${typography.fontSize['4xl']};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    font-size: ${typography.fontSize['3xl']};
  }
`;

export const H1 = styled.h1`
  ${headingStyles}
  font-size: ${typography.fontSize['4xl']};
  font-weight: 700;
  line-height: 1.2;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    font-size: ${typography.fontSize['3xl']};
  }
`;

export const H2 = styled.h2`
  ${headingStyles}
  font-size: ${typography.fontSize['3xl']};
  font-weight: 700;
  line-height: 1.25;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    font-size: ${typography.fontSize['2xl']};
  }
`;

export const H3 = styled.h3`
  ${headingStyles}
  font-size: ${typography.fontSize['2xl']};
  font-weight: 600;
  line-height: 1.3;
`;

export const H4 = styled.h4`
  ${headingStyles}
  font-size: ${typography.fontSize.xl};
  font-weight: 600;
  line-height: 1.4;
`;

export const H5 = styled.h5`
  ${headingStyles}
  font-size: ${typography.fontSize.lg};
  font-weight: 600;
  line-height: 1.4;
`;

export const H6 = styled.h6`
  ${headingStyles}
  font-size: ${typography.fontSize.base};
  font-weight: 600;
  line-height: 1.4;
`;

export const Text = styled.p`
  ${baseTextStyles}
  font-size: ${({ size }) => typography.fontSize[size] || typography.fontSize.base};
  font-weight: ${({ weight }) => typography.fontWeight[weight] || typography.fontWeight.normal};
  line-height: ${typography.lineHeight.relaxed};
  letter-spacing: -0.005em;
  color: ${({ theme, color, variant }) => {
    if (color) return color;
    if (variant === 'secondary') return theme.colors.text.secondary;
    if (variant === 'tertiary') return theme.colors.text.tertiary;
    return theme.colors.text.primary;
  }};
`;

export const SmallText = styled(Text)`
  font-size: ${typography.fontSize.sm};
  line-height: 1.5;
`;

export const Caption = styled(Text)`
  font-size: ${typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.tertiary};
`;

export const Label = styled.label`
  ${baseTextStyles}
  font-size: ${typography.fontSize.sm};
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text.secondary};
  display: block;
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

export const Link = styled.a`
  ${baseTextStyles}
  color: ${({ theme }) => theme.colors.primary[600]};
  text-decoration: none;
  cursor: pointer;
  transition: color ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.primary[700]};
    text-decoration: underline;
  }
`;

export default typography;
