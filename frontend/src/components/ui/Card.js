import styled, { css } from 'styled-components';

const cardVariants = {
  elevated: css`
    box-shadow: ${({ theme }) => theme.shadows.card};
  `,
  outlined: css`
    border: 1px solid ${({ theme }) => theme.colors.border.light};
  `,
  subtle: css`
    box-shadow: ${({ theme }) => theme.shadows.subtle};
  `,
  flat: css`
    background-color: ${({ theme }) => theme.colors.background.tertiary};
  `,
};

const Card = styled.div`
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-radius: ${({ theme, radius }) => theme.borderRadius[radius] || theme.borderRadius.xl};
  padding: ${({ theme, padding }) => theme.spacing[padding] || theme.spacing[5]};
  ${({ variant = 'elevated' }) => cardVariants[variant]}
`;

export const CardHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

export const CardTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.primary};
  margin: 0;
`;

export const CardDescription = styled.p`
  font-size: 0.875rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: ${({ theme }) => theme.spacing[1]} 0 0 0;
`;

export const CardContent = styled.div``;

export const CardFooter = styled.div`
  margin-top: ${({ theme }) => theme.spacing[4]};
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

export default Card;
