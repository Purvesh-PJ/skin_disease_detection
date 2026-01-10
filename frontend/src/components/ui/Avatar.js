import styled, { css } from 'styled-components';

const sizes = {
  sm: css`
    width: 28px;
    height: 28px;
  `,
  md: css`
    width: 36px;
    height: 36px;
  `,
  lg: css`
    width: 48px;
    height: 48px;
  `,
  xl: css`
    width: 64px;
    height: 64px;
  `,
};

const Avatar = styled.img`
  border-radius: 50%;
  object-fit: cover;
  background-color: ${({ theme }) => theme.colors.neutral[300]};
  ${({ size = 'md' }) => sizes[size]}
`;

export const AvatarFallback = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background-color: ${({ theme }) => theme.colors.primary[100]};
  color: ${({ theme }) => theme.colors.primary[700]};
  font-weight: 600;
  ${({ size = 'md' }) => sizes[size]}
  font-size: ${({ size }) => {
    switch (size) {
      case 'sm': return '0.75rem';
      case 'lg': return '1.25rem';
      case 'xl': return '1.5rem';
      default: return '1rem';
    }
  }};
`;

export default Avatar;
