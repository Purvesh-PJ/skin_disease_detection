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
  gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid;
  font-size: 0.875rem;
  font-weight: 500;
  line-height: 1.4;

  ${({ variant = 'info' }) => alertVariants[variant]}
`;

export default Alert;
