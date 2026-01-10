import styled from 'styled-components';
import { Input } from '../../components/common/ui';
import { SmallText } from '../../styles/typography';

export const Container = styled.div`
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
`;

export const AuthContainer = styled.div`
  display: flex;
  width: 100%;
  max-width: 1150px;
  min-height: 70vh;
  border-radius: ${({ theme }) => theme.borderRadius['3xl']};
  box-shadow: ${({ theme }) => theme.shadows.subtle};
  background-color: ${({ theme }) => theme.colors.background.primary};
  overflow: hidden;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    flex-direction: column;
    min-height: auto;
  }
`;

const Column = styled.div`
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[6]};
`;

export const LeftColumn = styled(Column)`
  background-color: ${({ theme }) => theme.colors.background.primary};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

export const RightColumn = styled(Column)`
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

export const Illustration = styled.div`
  width: 80%;
  text-align: center;

  img {
    width: 100%;
    height: auto;
  }
`;

export const Form = styled.form`
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-width: 400px;
  gap: ${({ theme }) => theme.spacing[4]};
`;

export const StyledInput = styled(Input)`
  width: 100%;
`;

export const LinkText = styled(SmallText)`
  text-align: center;
  color: ${({ theme }) => theme.colors.text.secondary};

  a {
    color: ${({ theme }) => theme.colors.primary[600]};
    text-decoration: none;
    font-weight: 500;

    &:hover {
      text-decoration: underline;
    }
  }
`;

export const Divider = styled.hr`
  width: 100%;
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  margin: ${({ theme }) => theme.spacing[2]} 0;
`;
