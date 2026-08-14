import React from 'react';
import styled, { css } from 'styled-components';
import { Slot } from '@radix-ui/react-slot';

const variants = {
  primary: css`
    background-color: ${({ theme }) => theme.colors.button.primary.bg};
    color: ${({ theme }) => theme.colors.button.primary.text};
    border: 1px solid transparent;
    box-shadow: ${({ theme }) => theme.shadows.sm};

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.primary.bgHover};
      box-shadow: ${({ theme }) => theme.shadows.md};
      transform: translateY(-1px);
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.primary.bgActive};
      transform: translateY(0);
    }
  `,
  secondary: css`
    background-color: ${({ theme }) => theme.colors.button.secondary.bg};
    color: ${({ theme }) => theme.colors.button.secondary.text};
    border: 1px solid ${({ theme }) => theme.colors.border.default};

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.secondary.bgHover};
      border-color: ${({ theme }) => theme.colors.border.dark};
      transform: translateY(-1px);
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.secondary.bgActive};
      transform: translateY(0);
    }
  `,
  accent: css`
    background-color: ${({ theme }) => theme.colors.primary[600]};
    color: #ffffff;
    border: 1px solid transparent;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25);

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.primary[700]};
      box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35);
      transform: translateY(-1px);
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.primary[800]};
      transform: translateY(0);
    }
  `,
  outline: css`
    background-color: transparent;
    color: ${({ theme }) => theme.colors.text.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.default};

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.hover};
      border-color: ${({ theme }) => theme.colors.border.dark};
      transform: translateY(-1px);
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.active};
      transform: translateY(0);
    }
  `,
  ghost: css`
    background-color: transparent;
    color: ${({ theme }) => theme.colors.text.secondary};
    border: 1px solid transparent;

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.hover};
      color: ${({ theme }) => theme.colors.text.primary};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.active};
    }
  `,
  danger: css`
    background-color: ${({ theme }) => theme.colors.error[600]};
    color: white;
    border: 1px solid transparent;

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.error[700]};
      transform: translateY(-1px);
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.error[800]};
      transform: translateY(0);
    }
  `,
};

const sizes = {
  sm: css`
    padding: ${({ theme }) => `${theme.spacing[1.5] || '6px'} ${theme.spacing[3]}`};
    font-size: 0.8125rem;
    gap: ${({ theme }) => theme.spacing[1.5] || '6px'};
  `,
  md: css`
    padding: ${({ theme }) => `${theme.spacing[2.5] || '10px'} ${theme.spacing[5]}`};
    font-size: 0.9375rem;
    gap: ${({ theme }) => theme.spacing[2]};
  `,
  lg: css`
    padding: ${({ theme }) => `${theme.spacing[3.5] || '14px'} ${theme.spacing[7] || '28px'}`};
    font-size: 1.05rem;
    gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  `,
};

const StyledButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: inherit;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  text-decoration: none;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  transition: all ${({ theme }) => theme.transitions.fast};

  ${({ variant = 'primary' }) => variants[variant]}
  ${({ size = 'md' }) => sizes[size]}
  ${({ fullWidth }) => fullWidth && css`width: 100%;`}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none !important;
    box-shadow: none !important;
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.primary[500]};
    outline-offset: 2px;
  }
`;

export const Button = React.forwardRef(({ asChild, ...props }, ref) => {
  const Component = asChild ? Slot : StyledButton;
  return <Component ref={ref} {...props} />;
});

Button.displayName = 'Button';

export default Button;
