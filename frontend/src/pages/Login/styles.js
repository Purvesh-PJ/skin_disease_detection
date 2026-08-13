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
