import styled, { css } from 'styled-components';

const alertVariants = {
  success: css`
    background-color: ${({ theme }) => theme.colors.status.success.bg};
    border-color: ${({ theme }) => theme.colors.status.success.border};
    color: ${({ theme }) => theme.colors.status.success.text};
  `,
  error: css`
    background-color: ${({ theme }) => theme.colors.status.error.bg};
    border-color: ${({ theme }) => theme.colors.status.error.border};
    color: ${({ theme }) => theme.colors.status.error.text};
  `,
  warning: css`
    background-color: ${({ theme }) => theme.colors.status.warning.bg};
    border-color: ${({ theme }) => theme.colors.status.warning.border};
    color: ${({ theme }) => theme.colors.status.warning.text};
  `,
  info: css`
    background-color: ${({ theme }) => theme.colors.status.info.bg};
    border-color: ${({ theme }) => theme.colors.status.info.border};
    color: ${({ theme }) => theme.colors.status.info.text};
  `,
};

const Alert = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid;
  font-size: 0.875rem;

  ${({ variant = 'info' }) => alertVariants[variant]}
`;

export default Alert;
