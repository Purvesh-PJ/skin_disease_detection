import styled, { keyframes } from 'styled-components';
import * as RadixDropdownMenu from '@radix-ui/react-dropdown-menu';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
`;

const StyledContent = styled(RadixDropdownMenu.Content)`
  min-width: 180px;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.card};
  padding: ${({ theme }) => theme.spacing[2]};
  z-index: 1000;
  animation: ${fadeIn} 0.15s ease-out;

  &:focus {
    outline: none;
  }
`;

const StyledItem = styled(RadixDropdownMenu.Item)`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: 0.875rem;
  cursor: pointer;
  outline: none;
  transition: background-color ${({ theme }) => theme.transitions.fast};

  &[data-highlighted] {
    background-color: ${({ theme }) => theme.colors.interactive.hover};
  }

  &.danger {
    color: ${({ theme }) => theme.colors.error[600]};
    
    &[data-highlighted] {
      background-color: ${({ theme }) => theme.colors.status.error.bg};
    }
  }
`;

const StyledSeparator = styled(RadixDropdownMenu.Separator)`
  height: 1px;
  background-color: ${({ theme }) => theme.colors.border.light};
  margin: ${({ theme }) => theme.spacing[2]} 0;
`;

export const DropdownMenuRoot = RadixDropdownMenu.Root;
export const DropdownMenuTrigger = RadixDropdownMenu.Trigger;
export const DropdownMenuContent = StyledContent;
export const DropdownMenuItem = StyledItem;
export const DropdownMenuSeparator = StyledSeparator;

const Dropdown = {
  Root: DropdownMenuRoot,
  Trigger: DropdownMenuTrigger,
  Content: DropdownMenuContent,
  Item: DropdownMenuItem,
  Separator: DropdownMenuSeparator,
};

export default Dropdown;
