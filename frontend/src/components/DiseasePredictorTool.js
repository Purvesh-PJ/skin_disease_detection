import styled from "styled-components";
import ImageUpload from "./ImageUpload";

const Container = styled.div`
    display : flex;
    flex-direction : row;
    width : 100%;
    max-width : 1200px;
    background-color : white;
    height : 750px;
    border-radius : 20px;
    box-sizing : border-box;
`;

const ImageUploadSection = styled.div`
    width : 50%;
    box-sizing : border-box;
`;

const ResultDisplaySection = styled.div`
    width : 50%;
    box-sizing : border-box;
    padding : 20px;
    margin : 5px;
    border : 2px dashed #ccc;
    border-radius: 20px;
`;

const Heading = styled.p`
    color : #475569;
    text-align : center;
`;

const DiseasePredictorTool = () => {

    return (
        <Container>
            <ImageUploadSection>
                <ImageUpload />
            </ImageUploadSection>
            <ResultDisplaySection>
                <Heading>Result Display</Heading>
            </ResultDisplaySection>
        </Container>
    )
}

export default DiseasePredictorTool;

