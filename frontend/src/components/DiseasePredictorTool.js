import styled from "styled-components";
import ImageUpload from "./ImageUpload";

const Container = styled.div`
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

`;

const DiseasePredictorTool = () => {

    return (
        <Container>
            <ImageUploadSection>
                <ImageUpload />
            </ImageUploadSection>
            <ResultDisplaySection>

            </ResultDisplaySection>
        
        </Container>
    )
}

export default DiseasePredictorTool;

