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
  gap: ${({ theme }) => theme.spacing[3]};
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

export const DemoNotice = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: ${({ theme }) => 
    theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(240, 253, 244, 0.8)'};
  border: 1px solid ${({ theme }) => 
    theme.mode === 'dark' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(187, 247, 208, 0.8)'};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  font-size: 0.78rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};

  span.bold {
    color: ${({ theme }) => theme.colors.primary[600] || '#16a34a'};
    font-weight: 700;
  }

  button.fill-btn {
    background: none;
    border: none;
    color: ${({ theme }) => theme.colors.primary[600] || '#16a34a'};
    font-weight: 700;
    cursor: pointer;
    text-decoration: underline;
    padding: 0;
    font-size: 0.78rem;
  }
`;

