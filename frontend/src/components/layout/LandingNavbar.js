import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { Button, ThemeToggle } from '../common/ui';
import { ROUTES } from '../../constants';
import { FiActivity, FiArrowRight } from 'react-icons/fi';

const NavHeader = styled.header`
  position: sticky;
  top: 0;
  left: 0;
  right: 0;
  z-index: 50;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  background-color: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(11, 15, 25, 0.88)' : 'rgba(255, 255, 255, 0.88)'};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  transition: all ${({ theme }) => theme.transitions.normal};
`;

const NavContainer = styled.div`
  max-width: 1320px;
  margin: 0 auto;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  display: flex;
  align-items: center;
  justify-content: space-between;

  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[4]}`};
  }
`;

const Brand = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  text-decoration: none;
`;

const BrandIcon = styled.div`
  width: 38px;
  height: 38px;
  background: ${({ theme }) => theme.colors.button.pine.bg};
  border: 1px solid ${({ theme }) => theme.colors.border.brand};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${({ theme }) => theme.colors.emerald.android};
  box-shadow: 0 4px 14px rgba(61, 220, 132, 0.2);
`;

const BrandName = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: ${({ theme }) => theme.colors.text.primary};

  span {
    color: ${({ theme }) => theme.colors.emerald.androidDark || '#16a34a'};
  }
`;

const NavLinks = styled.nav`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    display: none;
  }
`;

const NavLink = styled.a`
  font-size: 0.925rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.secondary};
  text-decoration: none;
  transition: color ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.emerald.androidDark || theme.colors.text.primary};
  }
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const LandingNavbar = ({ isAuthenticated }) => {
  return (
    <NavHeader>
      <NavContainer>
        <Brand to={ROUTES.HOME}>
          <BrandIcon>
            <FiActivity size={22} />
          </BrandIcon>
          <BrandName>Skin<span>AI</span></BrandName>
        </Brand>

        <NavLinks>
          <NavLink href="#overview">Overview</NavLink>
          <NavLink href="#sandbox">Live Sandbox</NavLink>
          <NavLink href="#architecture">Bento Engine</NavLink>
          <NavLink href="#models">AI Models</NavLink>
          <NavLink href="#conditions">Pathology Atlas</NavLink>
          <NavLink href="#workflow">Clinician Flow</NavLink>
        </NavLinks>

        <NavActions>
          <ThemeToggle />

          {isAuthenticated ? (
            <Button asChild variant="android" size="sm">
              <Link to={ROUTES.DASHBOARD}>
                Dashboard
                <FiArrowRight size={14} />
              </Link>
            </Button>
          ) : (
            <>
              <Button asChild variant="ghost" size="sm">
                <Link to={ROUTES.LOGIN}>Log in</Link>
              </Button>
              <Button asChild variant="android" size="sm">
                <Link to={ROUTES.SIGNUP}>
                  Get Started
                  <FiArrowRight size={14} />
                </Link>
              </Button>
            </>
          )}
        </NavActions>
      </NavContainer>
    </NavHeader>
  );
};

export default LandingNavbar;
