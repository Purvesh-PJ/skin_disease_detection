import React from 'react';
import '@testing-library/jest-dom';

// Robust window.matchMedia mock
window.matchMedia = window.matchMedia || function (query) {
  return {
    matches: false,
    media: query || '',
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  };
};

// Mock Radix UI components that hit Jest subpath exports issues in create-react-app
jest.mock('@radix-ui/react-tooltip', () => ({
  Provider: ({ children }) => <>{children}</>,
  Root: ({ children }) => <>{children}</>,
  Trigger: ({ children }) => <>{children}</>,
  Portal: ({ children }) => <>{children}</>,
  Content: ({ children }) => <>{children}</>,
  Arrow: () => null,
}));

jest.mock('@radix-ui/react-dropdown-menu', () => ({
  Root: ({ children }) => <>{children}</>,
  Trigger: ({ children }) => <>{children}</>,
  Content: ({ children }) => <>{children}</>,
  Item: ({ children, onClick }) => <div onClick={onClick}>{children}</div>,
  Separator: () => <hr />,
}));

jest.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children }) => <>{children}</>,
  Trigger: ({ children }) => <>{children}</>,
  Portal: ({ children }) => <>{children}</>,
  Overlay: () => null,
  Content: ({ children }) => <div>{children}</div>,
  Title: ({ children }) => <h2>{children}</h2>,
  Description: ({ children }) => <p>{children}</p>,
  Close: ({ children }) => <>{children}</>,
}));
