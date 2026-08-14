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
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  background-color: ${({ theme }) =>
    theme.mode === 'dark' ? 'rgba(15, 23, 42, 0.85)' : 'rgba(255, 255, 255, 0.85)'};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
  transition: all ${({ theme }) => theme.transitions.normal};
`;

const NavContainer = styled.div`
  max-width: 1280px;
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
  width: 36px;
  height: 36px;
  background: ${({ theme }) => theme.gradients.brandIcon};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
`;

const BrandName = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const NavLinks = styled.nav`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    display: none;
  }
`;

const NavLink = styled.a`
  font-size: 0.9rem;
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text.secondary};
  text-decoration: none;
  transition: color ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.text.primary};
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
            <FiActivity size={20} />
          </BrandIcon>
          <BrandName>Skin AI</BrandName>
        </Brand>

        <NavLinks>
          <NavLink href="#overview">Overview</NavLink>
          <NavLink href="#how-it-works">How It Works</NavLink>
          <NavLink href="#models">AI Models</NavLink>
          <NavLink href="#conditions">Conditions</NavLink>
          <NavLink href="#safety">Ethics & Safety</NavLink>
        </NavLinks>

        <NavActions>
          <ThemeToggle />

          {isAuthenticated ? (
            <Button asChild variant="primary" size="sm">
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
              <Button asChild variant="primary" size="sm">
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
