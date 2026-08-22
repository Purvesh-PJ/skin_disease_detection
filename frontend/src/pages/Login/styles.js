import styled from 'styled-components';
import { Input } from '../../components/common/ui';
import { SmallText } from '../../styles/typography';

export const FormHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[1]};
  text-align: center;
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

export const Form = styled.form`
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: ${({ theme }) => theme.spacing[4]};
`;

export const StyledInput = styled(Input)`
  width: 100%;
`;

export const LinkText = styled(SmallText)`
  text-align: center;
  color: ${({ theme }) => theme.colors.text.secondary};

  a {
    color: ${({ theme }) => theme.colors.primary[500]};
    font-weight: 600;

    &:hover {
      text-decoration: underline;
    }
  }
`;

export const Divider = styled.hr`
  width: 100%;
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  margin: ${({ theme }) => theme.spacing[1]} 0;
`;

export const DemoCard = styled.div`
  background: ${({ theme }) => 
    theme.mode === 'dark' 
      ? 'linear-gradient(135deg, rgba(22, 163, 74, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)' 
      : 'linear-gradient(135deg, rgba(220, 252, 231, 0.8) 0%, rgba(240, 253, 244, 0.5) 100%)'};
  border: 1px solid ${({ theme }) => theme.colors.primary[500] || '#16a34a'};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  padding: ${({ theme }) => theme.spacing[4]};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
  box-shadow: 0 4px 16px rgba(22, 163, 74, 0.15);
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

export const DemoHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;

  h4 {
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0;
    color: ${({ theme }) => theme.colors.text.primary};
    display: flex;
    align-items: center;
    gap: 6px;
  }

  span.badge {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: ${({ theme }) => theme.colors.primary[600]};
    color: #ffffff;
    padding: 2px 8px;
    border-radius: 9999px;
  }
`;

export const DemoDesc = styled(SmallText)`
  font-size: 0.8rem;
  line-height: 1.4;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin: 0;
`;

export const DemoButton = styled.button`
  width: 100%;
  padding: 10px 16px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: none;
  background: linear-gradient(135deg, #16a34a 0%, #059669 100%);
  color: #ffffff;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: 0 4px 12px rgba(22, 163, 74, 0.35);
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(22, 163, 74, 0.45);
    background: linear-gradient(135deg, #15803d 0%, #047857 100%);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

export const OrDivider = styled.div`
  display: flex;
  align-items: center;
  text-align: center;
  margin: ${({ theme }) => theme.spacing[2]} 0;
  color: ${({ theme }) => theme.colors.text.tertiary || '#888'};
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;

  &::before,
  &::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
  }

  span {
    padding: 0 ${({ theme }) => theme.spacing[3]};
  }
`;

