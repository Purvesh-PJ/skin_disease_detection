import React, { useState } from 'react';
import { UploadContainer, ImagePlaceholder, FileInput, UploadButton, ImagePreview, Image, Note, Paragraph } from './ImageUpload_Styles';
import useSkinDiseasePrediction from '../hooks/useSkinDiseasePrediction';
import Disease_icon from '../resources/icons/disease_icon.png';
import styled from "styled-components";
import { FiUpload } from 'react-icons/fi';

const Container = styled.div`
    display: flex;
    flex-direction: row;
    width: 100%;
    max-width: 1200px;
    background-color: white;
    min-height: 600px;
    max-height: calc(100vh - 100px);
    border-radius: 20px;
    box-sizing: border-box;
    box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;
    overflow: auto;
    
    @media (max-width: 1024px) {
        flex-direction: column;
        max-height: none;
        height: auto;
    }
    
    @media (max-width: 768px) {
        border-radius: 10px;
    }
`;

const ImageUploadSection = styled.div`
    width: 50%;
    box-sizing: border-box;
    
    @media (max-width: 1024px) {
        width: 100%;
    }
`;

const ResultDisplaySection = styled.div`
    width: 50%;
    box-sizing: border-box;
    padding: 20px;
    margin: 5px;
    border: 2px dashed #ccc;
    border-radius: 20px;
    text-align: center;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    
    @media (max-width: 1024px) {
        width: 100%;
        margin: 10px 5px;
    }
    
    @media (max-width: 768px) {
        border-radius: 10px;
    }
`;

const Heading = styled.p`
    color: #475569;
    text-align: center;
`;

const Loader = styled.div`
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    margin: 30px auto;
    animation: spin 1.5s linear infinite;

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;

const LoadingText = styled.p`
    color: #3b82f6;
    font-size: 18px;
    font-weight: 500;
    margin: 15px 0;
    animation: pulse 1.5s infinite;
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
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

const ResultText = styled.p`
    color: #4b5563;
    font-size: 16px;
    margin: 8px 0;
`;

const SuccessMessage = styled.div`
    color: #16a34a;
    font-size: 14px;
    padding: 8px;
    margin: 0 0 16px 0;
    border-radius: 5px;
    background-color: #f0fdf4;
    border: 1px solid #86efac;
    display: flex;
    align-items: center;
    justify-content: center;
`;

const ResultContainer = styled.div`
    margin-top: 20px;
`;

const ResultCard = styled.div`
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
`;

const ResultLabel = styled.span`
    // font-weight: 400;
    color: #334155;
`;

const ResultValue = styled.span`
    color: #4b5563;
`;

const ResultSection = styled.div`
    margin-bottom: 12px;
`;

const ResultTitle = styled.span`
    color:rgb(114, 114, 114);
    margin-bottom: 16px;
    font-size: 16px;
    // margin-left: auto;
    // margin-right: auto;
    background-color:rgba(219, 219, 219, 0.55);
    padding: 4px;
    border-radius: 8px;
    border : 1px solid lightgray;
`;

const NestedResultContainer = styled.div`
    margin-left: 20px;
    margin-top: 8px;
`;

const DiseaseDescription = styled.p`
    color:rgb(87, 87, 87);
    font-size: 16px;
    line-height: 1.5;
    margin: 12px 0;
    font-style: italic;
`;

const DiseaseName = styled.h4`
    color:rgb(29, 29, 29);
    font-size: 20px;
    margin: 16px 0 8px 0;
`;

const MessageBox = styled.div`
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 40px 30px;
    margin: 40px auto;
    // margin-top : auto;
    // margin-bottom : auto;
    text-align: center;
    width: 80%;
    height: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    
    p {
        color: #64748b;
        font-size: 14px;
        margin: 10px 0 0 0;
        font-weight: 500;
    }
`;

const IconWrapper = styled.div`
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 15px;
    background-color: #f0f9ff;
    border: 2px solid #93c5fd;
    border-radius: 50%;
    color: #3b82f6;
    font-size: 18px;
`;

const UploadArrow = styled.div`
    position: relative;
    width: 24px;
    height: 24px;
    
    /* Vertical line */
    &::before {
        content: "";
        position: absolute;
        width: 3px;
        height: 18px;
        background-color: gray;
        left: 50%;
        transform: translateX(-50%);
        bottom: 0;
    }
    
    /* Arrow head */
    &::after {
        content: "";
        position: absolute;
        width: 12px;
        height: 12px;
        border-top: 3px solid gray;
        border-left: 3px solid gray;
        transform: translateX(-50%) rotate(45deg);
        left: 50%;
        top: 0;
    }
`;

const DiseasePredictorTool = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [predictionResult, setPredictionResult] = useState(null);
    const [statusMessage, setStatusMessage] = useState(null);

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
        setStatusMessage(null);

        try {
            const response = await postImageToPredict(formData);
            if (response?.data) {
                setPredictionResult(response.data); // Assuming API sends result in `data`
                
                // Extract status message if available
                if (response.data.message) {
                    setStatusMessage(response.data.message);
                } else {
                    setStatusMessage("Analysis completed successfully");
                }
            } else {
                setPredictionResult(null);
                setStatusMessage(null);
                console.error("Unexpected response structure:", response);
            }
        } 
        catch (err) {
            console.error("Error during prediction:", err);
            setStatusMessage(null);
        }
    };

    // Helper function to get full disease name if available
    const getFullDiseaseName = (result) => {
        // Check if disease_details exists and has a name property
        if (result.disease_details && result.disease_details.name) {
            return result.disease_details.name;
        }
        // Otherwise return the predicted_disease value
        return result.predicted_disease;
    };

    // Helper function to get disease description if available
    const getDiseaseDescription = (result) => {
        // Check if disease_details exists and has a description property
        if (result.disease_details && result.disease_details.description) {
            return result.disease_details.description;
        }
        return null;
    };

    // Helper function to render all result data
    const renderResultData = (data) => {
        // Skip rendering these fields as they're handled separately or not needed for display
        const skipFields = ['predicted_disease', 'confidence', 'disease_details', 'message'];
        
        return Object.entries(data).map(([key, value]) => {
            // Skip the fields we're handling separately
            if (skipFields.includes(key)) return null;
            
            // Format the key for display (convert snake_case to Title Case)
            const formattedKey = key
                .split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            
            // Handle different value types
            if (typeof value === 'object' && value !== null) {
                // For objects, render each property separately
                return (
                    <ResultSection key={key}>
                        <ResultLabel>{formattedKey}: </ResultLabel>
                        <NestedResultContainer>
                            {Object.entries(value).map(([nestedKey, nestedValue]) => {
                                // Skip description as it's handled separately
                                if (nestedKey === 'description' || nestedKey === 'name' || nestedKey === 'message') return null;
                                
                                // Format nested key
                                const formattedNestedKey = nestedKey
                                    .split('_')
                                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                    .join(' ');
                                
                                return (
                                    <ResultSection key={`${key}-${nestedKey}`}>
                                        <ResultLabel>{formattedNestedKey}: </ResultLabel>
                                        <ResultValue>{nestedValue}</ResultValue>
                                    </ResultSection>
                                );
                            })}
                        </NestedResultContainer>
                    </ResultSection>
                );
            } else if (Array.isArray(value)) {
                // For arrays, join the values with commas
                return (
                    <ResultSection key={key}>
                        <ResultLabel>{formattedKey}: </ResultLabel>
                        <ResultValue>{value.join(', ')}</ResultValue>
                    </ResultSection>
                );
            } else {
                // For simple values
                return (
                    <ResultSection key={key}>
                        <ResultLabel>{formattedKey}: </ResultLabel>
                        <ResultValue>{value}</ResultValue>
                    </ResultSection>
                );
            }
        });
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
                        disabled={loading}
                    />
                    <UploadButton 
                        onClick={handleUploadClick} 
                        disabled={loading || !imageFile}
                    >
                        {loading ? "Processing..." : "Upload"}
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
                {loading ? (
                    <ResultContainer>
                        <Loader />
                        <LoadingText>Analyzing image...</LoadingText>
                        <ResultText>Please wait while we process your image</ResultText>
                    </ResultContainer>
                ) : predictionResult ? (
                    <ResultContainer>
                        <ResultCard>
                            {statusMessage && (
                                <SuccessMessage>{statusMessage}</SuccessMessage>
                            )}
                            <ResultTitle>Predicted disease</ResultTitle>
                            
                            {/* Display full disease name */}
                            <DiseaseName>{getFullDiseaseName(predictionResult)}</DiseaseName>
                            
                            {/* Display disease description if available */}
                            {getDiseaseDescription(predictionResult) && (
                                <DiseaseDescription>
                                    {getDiseaseDescription(predictionResult)}
                                </DiseaseDescription>
                            )}
                            
                            <ResultSection>
                                <ResultLabel>Confidence: </ResultLabel>
                                <ResultValue>{predictionResult.confidence}%</ResultValue>
                            </ResultSection>
                            
                            {/* Render all other fields from the response */}
                            {renderResultData(predictionResult)}
                        </ResultCard>
                    </ResultContainer>
                ) : (
                    <MessageBox>
                        <IconWrapper>
                            <FiUpload />
                        </IconWrapper>
                        <p>Upload an image to see prediction results.</p>
                    </MessageBox>
                )}
            </ResultDisplaySection>
        </Container>
    );
};

export default DiseasePredictorTool;


