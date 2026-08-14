import styled, { css } from 'styled-components';

const cardVariants = {
  elevated: css`
    background-color: ${({ theme }) => theme.colors.background.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.default};
    box-shadow: ${({ theme }) => theme.shadows.paper};
  `,
  outlined: css`
    background-color: ${({ theme }) => theme.colors.background.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.default};
    box-shadow: none;
  `,
  subtle: css`
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    border: 1px solid ${({ theme }) => theme.colors.border.light};
    box-shadow: none;
  `,
  flat: css`
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    border: none;
    box-shadow: none;
  `,
  interactive: css`
    background-color: ${({ theme }) => theme.colors.background.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.default};
    box-shadow: ${({ theme }) => theme.shadows.paper};
    transition: all ${({ theme }) => theme.transitions.fast};

    &:hover {
      border-color: ${({ theme }) => theme.colors.primary[400]};
      box-shadow: ${({ theme }) => theme.shadows.hover};
      transform: translateY(-2px);
    }
  `,
};

const Card = styled.div`
  border-radius: ${({ theme, radius }) => (radius && theme.borderRadius[radius]) || theme.borderRadius.card};
  padding: ${({ theme, padding }) => (padding && theme.spacing[padding]) || theme.spacing[6]};
  ${({ variant = 'elevated' }) => cardVariants[variant]}
`;

export default Card;
