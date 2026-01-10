import styled from 'styled-components';

export const DropdownMenu = styled.ul`
  position: absolute;
  top: 100%;
  right: 0;
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  margin-top: ${({ theme }) => theme.spacing[2]};
  list-style: none;
  padding: ${({ theme }) => theme.spacing[1]};
  min-width: 120px;
  z-index: 1000;
  box-shadow: ${({ theme }) => theme.shadows.lg};
`;

export const DropdownItem = styled.li`
  padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
  cursor: pointer;
  font-size: 0.875rem;
  color: ${({ theme }) => theme.colors.text.primary};
  border-radius: ${({ theme }) => theme.borderRadius.sm};
  transition: background-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.neutral[100]};
  }
`;

const Dropdown = { DropdownMenu, DropdownItem };
export default Dropdown;
