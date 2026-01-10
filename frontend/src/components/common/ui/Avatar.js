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

export default Avatar;
