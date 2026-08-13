import React from 'react';
import styled, { css } from 'styled-components';
import { Slot } from '@radix-ui/react-slot';

const variants = {
  primary: css`
    background-color: ${({ theme }) => theme.colors.button.primary.bg};
    color: ${({ theme }) => theme.colors.button.primary.text};
    border: none;

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.primary.bgHover};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.primary.bgActive};
    }
  `,
  secondary: css`
    background-color: ${({ theme }) => theme.colors.button.secondary.bg};
    color: ${({ theme }) => theme.colors.button.secondary.text};
    border: 1px solid ${({ theme }) => theme.colors.button.secondary.border};

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.secondary.bgHover};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.button.secondary.bgActive};
    }
  `,
  outline: css`
    background-color: transparent;
    color: ${({ theme }) => theme.colors.primary[600]};
    border: 2px solid ${({ theme }) => theme.colors.primary[600]};

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.selected};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.selectedHover};
    }
  `,
  ghost: css`
    background-color: transparent;
    color: ${({ theme }) => theme.colors.text.primary};
    border: none;

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.hover};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.interactive.active};
    }
  `,
  danger: css`
    background-color: ${({ theme }) => theme.colors.error[600]};
    color: white;
    border: none;

    &:hover:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.error[700]};
    }

    &:active:not(:disabled) {
      background-color: ${({ theme }) => theme.colors.error[800]};
    }
  `,
};

const sizes = {
  sm: css`
    padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[3]}`};
    font-size: 0.875rem;
    border-radius: ${({ theme }) => theme.borderRadius.sm};
  `,
  md: css`
    padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[4]}`};
    font-size: 1rem;
    border-radius: ${({ theme }) => theme.borderRadius.md};
  `,
  lg: css`
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
    font-size: 1.125rem;
    border-radius: ${({ theme }) => theme.borderRadius.md};
  `,
};

const StyledButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing[2]};
  font-weight: 500;
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  white-space: nowrap;

  ${({ variant = 'primary' }) => variants[variant]}
  ${({ size = 'md' }) => sizes[size]}
  ${({ fullWidth }) => fullWidth && css`width: 100%;`}

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
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
