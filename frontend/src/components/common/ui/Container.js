import styled, { css } from 'styled-components';

const Container = styled.div`
  width: 100%;
  margin-left: auto;
  margin-right: auto;
  padding-left: ${({ theme }) => theme.spacing[4]};
  padding-right: ${({ theme }) => theme.spacing[4]};

  ${({ maxWidth }) => {
    switch (maxWidth) {
      case 'sm':
        return css`max-width: 640px;`;
      case 'md':
        return css`max-width: 768px;`;
      case 'lg':
        return css`max-width: 1024px;`;
      case 'xl':
        return css`max-width: 1280px;`;
      case 'full':
        return css`max-width: 100%;`;
      default:
        return css`max-width: 1200px;`;
    }
  }}
`;

export default Container;
