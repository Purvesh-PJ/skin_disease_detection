import styled from 'styled-components';


export const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: auto;
  min-height: 600px;
  max-height: calc(100vh - 120px);
  margin: 5px;
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 20px;
  box-sizing: border-box;
  overflow-y: auto;
  
  @media (max-width: 1024px) {
    min-height: 500px;
  }
  
  @media (max-width: 768px) {
    padding: 15px;
    border-radius: 10px;
  }
`;

export const ImagePlaceholder = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 300px;
  background-color: #f1f5f9;
  border-radius: 20px;
  margin-bottom: 15px;
  font-size: 2em;
  color: #aaa;
  
  @media (max-width: 768px) {
    height: 250px;
    border-radius: 10px;
  }
  
  @media (max-width: 480px) {
    height: 200px;
  }
`;

export const Image = styled.img`
  width: 120px;
  height: 120px;
  
  @media (max-width: 768px) {
    width: 100px;
    height: 100px;
  }
  
  @media (max-width: 480px) {
    width: 80px;
    height: 80px;
  }
`;

export const FileInput = styled.input`
  margin-top: 10px;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
  border: 2px dashed gray;
  border-radius: 5px;
  background-color: white;
  
  &:disabled {
    background-color: #f3f4f6;
    border-color: #d1d5db;
    cursor: not-allowed;
    opacity: 0.7;
  }
`;

export const UploadButton = styled.button`
  margin-top: 15px;
  padding: 10px 20px;
  background-color: black;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  width: 100%;
  transition: all 0.3s ease;
  
  &:hover:not(:disabled) {
    background-color: #f1f5f9;
    color: black;
  }
  
  &:disabled {
    background-color: #d1d5db;
    color: #6b7280;
    cursor: not-allowed;
  }
`;

export const ImagePreview = styled.img`
  width: 100%;
  height: 300px;
  border-radius: 10px;
  // object-fit: cover;
  // margin-bottom: 15px;
  
  @media (max-width: 768px) {
    height: 250px;
  }
  
  @media (max-width: 480px) {
    height: 200px;
  }
`;

export const Note = styled.p`
  margin-top: 15px;
  font-size: 0.8em;
  color: #666;
  text-align: center;
  background-color: #f3f4f6;
  padding: 10px;
  border-radius: 8px;
`;

export const Paragraph = styled.p`
  color: #475569;
  text-align: center;
`;
