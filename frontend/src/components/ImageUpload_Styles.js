import styled from 'styled-components';

export const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: auto;
  min-height: 600px;
  max-height: calc(100vh - 120px);
  margin: ${({ theme }) => theme.spacing[1]};
  padding: ${({ theme }) => theme.spacing[5]};
  border: 2px dashed ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  box-sizing: border-box;
  overflow-y: auto;
  background-color: ${({ theme }) => theme.colors.background.primary};

  @media (max-width: ${({ theme }) => theme.breakpoints.lg}) {
    min-height: 500px;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    padding: ${({ theme }) => theme.spacing[4]};
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }
`;

export const ImagePlaceholder = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 300px;
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
  font-size: 2em;
  color: ${({ theme }) => theme.colors.text.tertiary};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    height: 250px;
    border-radius: ${({ theme }) => theme.borderRadius.lg};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    height: 200px;
  }
`;

export const Image = styled.img`
  width: 120px;
  height: 120px;

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    width: 100px;
    height: 100px;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    width: 80px;
    height: 80px;
  }
`;

export const FileInput = styled.input`
  margin-top: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => theme.spacing[2]};
  width: 100%;
  box-sizing: border-box;
  border: 2px dashed ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  background-color: ${({ theme }) => theme.colors.background.primary};
  color: ${({ theme }) => theme.colors.text.primary};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    border-color: ${({ theme }) => theme.colors.primary[400]};
    background-color: ${({ theme }) => theme.colors.interactive.selected};
  }

  &:disabled {
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    border-color: ${({ theme }) => theme.colors.border.light};
    cursor: not-allowed;
    opacity: 0.7;
  }
`;

export const UploadButton = styled.button`
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[5]}`};
  background-color: ${({ theme }) => theme.colors.primary[600]};
  color: white;
  border: none;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  cursor: pointer;
  font-size: 1em;
  font-weight: 500;
  width: 100%;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    background-color: ${({ theme }) => theme.colors.primary[700]};
  }

  &:disabled {
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    color: ${({ theme }) => theme.colors.text.tertiary};
    cursor: not-allowed;
  }
`;

export const ImagePreview = styled.img`
  width: 100%;
  height: 300px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  object-fit: contain;
  background-color: ${({ theme }) => theme.colors.background.tertiary};

  @media (max-width: ${({ theme }) => theme.breakpoints.md}) {
    height: 250px;
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    height: 200px;
  }
`;

export const Note = styled.p`
  margin-top: ${({ theme }) => theme.spacing[4]};
  font-size: 0.8em;
  color: ${({ theme }) => theme.colors.status.warning.text};
  text-align: center;
  background-color: ${({ theme }) => theme.colors.status.warning.bg};
  padding: ${({ theme }) => theme.spacing[3]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  border: 1px solid ${({ theme }) => theme.colors.status.warning.border};
`;

export const Paragraph = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;
  font-weight: 500;
`;
