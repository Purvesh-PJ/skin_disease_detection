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
    theme.mode === 'dark' ? 'rgba(15, 23, 42, 0.9)' : 'rgba(255, 255, 255, 0.9)'};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
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
  background: ${({ theme }) => theme.colors.primary[600]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 2px 8px rgba(22, 163, 74, 0.25);
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

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
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
    color: ${({ theme }) => theme.colors.primary[600]};
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
          <BrandName>Skin Disease AI</BrandName>
        </Brand>

        <NavLinks>
          <NavLink href="#overview">Overview</NavLink>
          <NavLink href="#pipeline">AI Pipeline</NavLink>
          <NavLink href="#dataset">HAM10000 Dataset</NavLink>
          <NavLink href="#models">Ensemble Models</NavLink>
          <NavLink href="#conditions">7 Conditions</NavLink>
          <NavLink href="#disclaimer">Project Notice</NavLink>
        </NavLinks>

        <NavActions>
          <ThemeToggle />

          {isAuthenticated ? (
            <Button asChild variant="brand" size="sm">
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
              <Button asChild variant="brand" size="sm">
                <Link to={ROUTES.DASHBOARD}>
                  Try Predictor
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
