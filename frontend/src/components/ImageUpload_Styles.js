import styled from 'styled-components';


export const UploadContainer = styled.div`
  display: flex;
  flex-direction: column;
  // justify-content: space-around;
  margin: 5px;
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 10px;
  background-color: #f9f9f9;
  box-sizing : border-box;
  width : 100%;
`;

export const ImagePlaceholder = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  max-with : 250px;
  height: 400px;
  background-color: #e0e0e0;
  border-radius: 10px;
  margin-bottom: 15px;
  font-size: 2em;
  color: #aaa;
`;

export const FileInput = styled.input`
  margin-top: 10px;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
  border: 2px dashed #64748b;
  border-radius: 5px;
  background-color : white;
`;

export const UploadButton = styled.button`
  margin-top: 15px;
  padding: 10px 20px;
  background-color: #6c63ff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  width: 100%;
  
  &:hover {
    background-color: #5753c9;
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
`;
