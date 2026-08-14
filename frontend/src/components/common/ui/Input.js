import styled, { css } from 'styled-components';

const inputSizes = {
  sm: css`
    padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
    font-size: 0.875rem;
    border-radius: ${({ theme }) => theme.borderRadius.md};
  `,
  md: css`
    padding: ${({ theme }) => `${theme.spacing[2.5] || '10px'} ${theme.spacing[4]}`};
    font-size: 0.95rem;
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  `,
  lg: css`
    padding: ${({ theme }) => `${theme.spacing[3.5] || '14px'} ${theme.spacing[4.5] || '18px'}`};
    font-size: 1rem;
    border-radius: ${({ theme }) => theme.borderRadius.xl};
  `,
};

const Input = styled.input`
  width: 100%;
  box-sizing: border-box;
  font-family: inherit;
  border: 1px solid ${({ theme, error }) => 
    error ? theme.colors.status.error.border : theme.colors.border.default};
  background-color: ${({ theme }) => theme.colors.background.primary};
  color: ${({ theme }) => theme.colors.text.primary};
  transition: all ${({ theme }) => theme.transitions.fast};
  outline: none;

  ${({ size = 'md' }) => inputSizes[size]}

  &::placeholder {
    color: ${({ theme }) => theme.colors.text.tertiary};
  }

  &:hover:not(:disabled) {
    border-color: ${({ theme, error }) => 
      error ? theme.colors.error[600] : theme.colors.border.dark};
  }

  &:focus {
    border-color: ${({ theme, error }) => 
      error ? theme.colors.status.error.border : theme.colors.primary[500]};
    box-shadow: 0 0 0 3px ${({ theme, error }) => 
      error ? 'rgba(239, 68, 68, 0.15)' : 'rgba(14, 165, 233, 0.15)'};
  }

  &:disabled {
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    border-color: ${({ theme }) => theme.colors.border.light};
    cursor: not-allowed;
    opacity: 0.6;
  }
`;

export default Input;
