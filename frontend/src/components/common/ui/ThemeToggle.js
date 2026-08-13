import styled from 'styled-components';
import { FiSun, FiMoon } from 'react-icons/fi';
import { useTheme } from '../../../context/ThemeContext';
import Tooltip from './Tooltip';

const ToggleButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: ${({ theme }) => theme.borderRadius.full};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  color: ${({ theme }) => theme.colors.text.secondary};
  cursor: pointer;
  outline: none;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.background.primary};
    color: ${({ theme }) => theme.colors.primary[500]};
  }

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.primary[500]};
    outline-offset: 2px;
  }
`;

const ThemeToggle = () => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <Tooltip content={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
      <ToggleButton onClick={toggleTheme} aria-label="Toggle theme">
        {isDarkMode ? <FiSun size={18} /> : <FiMoon size={18} />}
      </ToggleButton>
    </Tooltip>
  );
};

export default ThemeToggle;
