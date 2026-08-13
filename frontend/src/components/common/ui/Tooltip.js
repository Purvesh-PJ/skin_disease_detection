import React from 'react';
import styled, { keyframes } from 'styled-components';
import * as RadixTooltip from '@radix-ui/react-tooltip';

const fadeIn = keyframes`
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
`;

const StyledContent = styled(RadixTooltip.Content)`
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: ${({ theme }) => `${theme.spacing[1]} ${theme.spacing[3]}`};
  font-size: 0.75rem;
  font-weight: 500;
  color: ${({ theme }) => theme.colors.text.inverse};
  background-color: ${({ theme }) => theme.colors.text.primary};
  box-shadow: ${({ theme }) => theme.shadows.md};
  animation: ${fadeIn} 0.15s ease-out;
  z-index: 1000;
`;

const StyledArrow = styled(RadixTooltip.Arrow)`
  fill: ${({ theme }) => theme.colors.text.primary};
`;

export const TooltipProvider = RadixTooltip.Provider;

export const Tooltip = ({ children, content, side = 'top', align = 'center', delayDuration = 200 }) => {
  if (!content) return children;

  return (
    <RadixTooltip.Provider delayDuration={delayDuration}>
      <RadixTooltip.Root>
        <RadixTooltip.Trigger asChild>
          {children}
        </RadixTooltip.Trigger>
        <RadixTooltip.Portal>
          <StyledContent side={side} align={align} sideOffset={5}>
            {content}
            <StyledArrow />
          </StyledContent>
        </RadixTooltip.Portal>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  );
};

export default Tooltip;
