import React, { useState } from 'react';
import { UploadContainer, ImagePlaceholder, FileInput, UploadButton, ImagePreview, Image, Note, Paragraph } from './ImageUpload_Styles';
import useSkinDiseasePrediction from '../hooks/useSkinDiseasePrediction';
import Disease_icon from '../resources/icons/disease_icon.png';
import styled from "styled-components";

const Container = styled.div`
    display: flex;
    flex-direction: row;
    width: 100%;
    max-width: 1200px;
    background-color: white;
    height: 750px;
    border-radius: 20px;
    box-sizing: border-box;
    box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px;
`;

const ImageUploadSection = styled.div`
    width: 50%;
    box-sizing: border-box;
`;

const ResultDisplaySection = styled.div`
    width: 50%;
    box-sizing: border-box;
    padding: 20px;
    margin: 5px;
    border: 2px dashed #ccc;
    border-radius: 20px;
    text-align: center;
`;

const Heading = styled.p`
    color: #475569;
    text-align: center;
`;

const Loader = styled.div`
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    margin: 20px auto;
    animation: spin 2s linear infinite;

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;

const ErrorMessage = styled.div`
    color: #ef4444;
    font-size: 12px;
    padding : 8px;
    margin : 8px;
    border-radius : 5px;
    background-color : #fef2f2;
    border : 1px solid #f87171;
`;

const DiseasePredictorTool = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [predictionResult, setPredictionResult] = useState(null);

    const { postImageToPredict, error, loading } = useSkinDiseasePrediction();

    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedImage(URL.createObjectURL(file));
            setImageFile(file);
            setPredictionResult(null); // Reset previous predictions
        } 
        else {
            setSelectedImage(null);
            setImageFile(null);
        }
    };

    const handleUploadClick = async () => {
        if (!imageFile) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", imageFile);

        try {
            const response = await postImageToPredict(formData);
            if (response?.data) {
                setPredictionResult(response.data); // Assuming API sends result in `data`
            } else {
                setPredictionResult(null);
                console.error("Unexpected response structure:", response);
            }
        } 
        catch (err) {
            console.error("Error during prediction:", err);
        }
    };

    return (
        <Container>
            <ImageUploadSection>
                <UploadContainer>
                    <Paragraph>Upload image</Paragraph>
                    {selectedImage ? (
                        <ImagePreview src={selectedImage} alt="Preview" />
                    ) : (
                        <ImagePlaceholder>
                            <Image src={Disease_icon} alt="disease-image" />
                        </ImagePlaceholder>
                    )}
                    <FileInput
                        type="file"
                        accept="image/*"
                        onChange={handleImageChange}
                        placeholder="Select an image"
                    />
                    <UploadButton onClick={handleUploadClick} disabled={loading}>
                        {loading ? "Uploading..." : "Upload"}
                    </UploadButton>
                    {error && <ErrorMessage><strong>error : </strong>{error.response?.data?.message || "An error occurred while processing the image."}</ErrorMessage>}
                    <Note>
                        <strong>WARNING:</strong> This model is trained only for skin disease images. Other images may not work.
                    </Note>
                </UploadContainer>
            </ImageUploadSection>

            <ResultDisplaySection>
                <Heading>
                    Result Display
                </Heading>
                {predictionResult ? (
                    <div>
                        <p><strong>Predicted Disease:</strong> {predictionResult.prediction}</p>
                        <p><strong>Confidence:</strong> {predictionResult.confidence}%</p>
                    </div>
                ) : (
                    <p>No results to display.</p>
                )}
            </ResultDisplaySection>
        </Container>
    );
};

export default DiseasePredictorTool;


