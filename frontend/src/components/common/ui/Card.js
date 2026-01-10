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

export default Card;
