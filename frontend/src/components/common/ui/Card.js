import styled, { css } from 'styled-components';

const cardVariants = {
  elevated: css`
    background-color: ${({ theme }) => theme.colors.background.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.light};
    box-shadow: ${({ theme }) => theme.shadows.paper};
  `,
  bento: css`
    background-color: ${({ theme }) => theme.colors.background.primary};
    border: 1px solid ${({ theme }) => theme.colors.border.light};
    border-radius: ${({ theme }) => theme.borderRadius.bento};
    box-shadow: ${({ theme }) => theme.shadows.bento};
    transition: all ${({ theme }) => theme.transitions.normal};

    &:hover {
      border-color: ${({ theme }) => theme.colors.border.brand};
      box-shadow: ${({ theme }) => theme.shadows.hover};
      transform: translateY(-2px);
    }
  `,
  bentoPine: css`
    background: ${({ theme }) => theme.gradients.bentoPine};
    border: 1px solid rgba(61, 220, 132, 0.2);
    border-radius: ${({ theme }) => theme.borderRadius.bento};
    box-shadow: ${({ theme }) => theme.shadows.xl};
    color: white;
  `,
  tonalMint: css`
    background: ${({ theme }) => theme.colors.background.tonalMint};
    border: 1px solid ${({ theme }) => theme.colors.border.brand};
    border-radius: ${({ theme }) => theme.borderRadius.bento};
  `,
  tonalIndigo: css`
    background: ${({ theme }) => theme.colors.background.tonalIndigo};
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: ${({ theme }) => theme.borderRadius.bento};
  `,
  tonalSand: css`
    background: ${({ theme }) => theme.colors.background.tonalSand};
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: ${({ theme }) => theme.borderRadius.bento};
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
};

const Card = styled.div`
  border-radius: ${({ theme, radius }) => (radius && theme.borderRadius[radius]) || theme.borderRadius.card};
  padding: ${({ theme, padding }) => (padding && theme.spacing[padding]) || theme.spacing[6]};
  ${({ variant = 'elevated' }) => cardVariants[variant]}
`;

export default Card;
