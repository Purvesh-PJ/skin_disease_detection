import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const sizes = {
  sm: '16px',
  md: '24px',
  lg: '40px',
};

const Spinner = styled.div`
  width: ${({ size = 'md' }) => sizes[size]};
  height: ${({ size = 'md' }) => sizes[size]};
  border: 3px solid ${({ theme }) => theme.colors.neutral[200]};
  border-top-color: ${({ theme, color }) => color || theme.colors.primary[600]};
  border-radius: 50%;
  animation: ${spin} 0.8s linear infinite;
`;

export default Spinner;
