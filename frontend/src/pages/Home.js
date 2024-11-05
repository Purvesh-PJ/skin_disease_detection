import { Container, Section, ImageUploadSection, PredectedResultSection } from './Home_Styles';
import ImageUpload from '../components/ImageUpload';

const Home = () => {

    return(
        <Container>
            <h1>Skin Disease Predictor</h1>
            <Section>
                <ImageUploadSection>
                    <ImageUpload></ImageUpload>

                </ImageUploadSection>
                <PredectedResultSection>

                </PredectedResultSection>

            </Section>
        </Container>
    )
}

export default Home