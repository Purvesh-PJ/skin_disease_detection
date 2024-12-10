import React, { useState } from 'react';
import { UploadContainer, ImagePlaceholder, FileInput, UploadButton, ImagePreview, Image, Note, Paragraph  } from './ImageUpload_Styles';
import usePostImageForPrediction from '../hooks/usePostImageForPrediction';
import Disease_icon from '../resources/icons/disease_icon.png';


const ImageUpload = () => {

  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile ] = useState(null);
  const { postImageToPredict, error, loading } = usePostImageForPrediction();

  console.log(selectedImage);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      console.log(file);
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
    } 
    else {
      setSelectedImage(null);
      setImageFile(null);
    }
  };

  const handleUploadClick = async () => {
    if(!imageFile) {
      alert("Please select an image first!");
      return;
    }
    const formData = new FormData();
    formData.append("image", imageFile);
    const res = await postImageToPredict(formData);
    console.log(res);
    alert(`Image uploaded successfully!`);
  };

  return (
    <UploadContainer>

      <Paragraph> Upload image </Paragraph>

      {selectedImage ? (
        <ImagePreview src={selectedImage} alt="Preview" />
      ) : (
        <ImagePlaceholder>
          <Image src={Disease_icon} alt='disease-image'/>
        </ImagePlaceholder>
      )}

      <FileInput
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        placeholder="Select an image"
      />

      <UploadButton onClick={handleUploadClick}> Upload </UploadButton>

      <Note>
        <strong> WARNING: </strong> This model is trained only for skin disease images. Other images may not work.
      </Note>

    </UploadContainer>
  );
};

export default ImageUpload;
