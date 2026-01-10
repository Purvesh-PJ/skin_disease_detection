import styled, { css } from 'styled-components';

const inputSizes = {
  sm: css`
    padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[2]}`};
    font-size: 0.875rem;
  `,
  md: css`
    padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
    font-size: 1rem;
  `,
  lg: css`
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
    font-size: 1.125rem;
  `,
};

const Input = styled.input`
  width: 100%;
  border: 2px solid ${({ theme, error }) => 
    error ? theme.colors.status.error.border : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.md};
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
      error ? theme.colors.status.error.bg : theme.colors.interactive.selected};
  }

  &:disabled {
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    border-color: ${({ theme }) => theme.colors.border.light};
    cursor: not-allowed;
    opacity: 0.7;
  }
`;

export default Input;
