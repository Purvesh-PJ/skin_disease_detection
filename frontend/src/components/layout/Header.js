import styled from 'styled-components';
import { FiLogOut, FiUser, FiSettings } from 'react-icons/fi';
import { ThemeToggle } from '../common/ui';
import Dropdown from '../common/ui/Dropdown';
import { authService } from '../../services';
import DefaultProfile from '../../assets/images/default_profile.jpg';

const HeaderContainer = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
`;

const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const HeaderRight = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary[500]}, ${({ theme }) => theme.colors.primary[700]});
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: 0.9rem;
`;

const LogoText = styled.span`
  font-size: 1.1rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.primary};
  
  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    display: none;
  }
`;

const ProfileTrigger = styled.button`
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  outline: none;
`;

const ProfileImage = styled.img`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid ${({ theme }) => theme.colors.border.light};
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
  }
`;

const Header = () => {
  const user = authService.getUser();

  const handleLogout = () => {
    authService.logout();
  };

  return (
    <HeaderContainer>
      <HeaderLeft>
        <Logo>
          <LogoIcon>SP</LogoIcon>
          <LogoText>Skin Disease Predictor</LogoText>
        </Logo>
      </HeaderLeft>

      <HeaderRight>
        <ThemeToggle />
        
        <Dropdown.Root>
          <Dropdown.Trigger asChild>
            <ProfileTrigger aria-label="User account menu">
              <ProfileImage src={DefaultProfile} alt="Profile" />
            </ProfileTrigger>
          </Dropdown.Trigger>

          <Dropdown.Content align="end" sideOffset={8}>
            <Dropdown.Item>
              <FiUser size={14} />
              {user?.username || 'Profile'}
            </Dropdown.Item>
            <Dropdown.Item>
              <FiSettings size={14} />
              Settings
            </Dropdown.Item>
            <Dropdown.Separator />
            <Dropdown.Item className="danger" onClick={handleLogout}>
              <FiLogOut size={14} />
              Logout
            </Dropdown.Item>
          </Dropdown.Content>
        </Dropdown.Root>
      </HeaderRight>
    </HeaderContainer>
  );
};

export default Header;
