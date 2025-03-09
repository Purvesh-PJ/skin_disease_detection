import styled from 'styled-components';


export const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height : 740px;
  margin: 5px;
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 20px;
  box-sizing : border-box;
`;

export const ImagePlaceholder = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  max-with : 250px;
  height: 400px;
  background-color: #f1f5f9;
  border-radius: 20px;
  margin-bottom: 15px;
  font-size: 2em;
  color: #aaa;
`;

export const Image = styled.img`
  width : 120px;
  height : 120px;
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
  width : 100%;
  height : 400px;
  border-radius: 10px;
  object-fit: cover;
  margin-bottom: 15px;
`;

export const Note = styled.p`
  margin-top: 15px;
  font-size: 0.8em;
  color: #666;
  text-align: center;
  background-color: #f3f4f6;
  padding : 10px;
  border-radius : 8px;
`;

export const Paragraph = styled.p`
  color : #475569;
  text-align: center;
`;
