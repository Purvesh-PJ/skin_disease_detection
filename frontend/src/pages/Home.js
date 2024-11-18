import { LandingWrapper, HeroSection, FeaturesSection, FeatureCard, Footer , UL, LI, Container, Section, ImageUploadSection, PredectedResultSection } from '../pages/Home_Styles'
// import { Link } from "react-router-dom";
import ImageUpload from '../components/ImageUpload';


const Home = () => {

    return(
        <LandingWrapper>
      <HeroSection>
        <h1>Skin Disease Detection</h1>
        <p>
          A reliable, AI-powered tool to analyze dermoscopic images and provide
          accurate predictions for skin conditions.
        </p>
        {/* <Link to="/skin-disease-predictor">Get Started</Link> */}
      </HeroSection>

      <FeaturesSection>
        <FeatureCard>
          <h2>About the Project</h2>
          <p>
            This application leverages advanced machine learning to identify
            skin conditions, assisting both dermatologists and individuals in
            early detection and diagnosis.
          </p>
        </FeatureCard>

        <FeatureCard>
          <h2>How It Works</h2>
          <UL>
            <LI>Upload a clear image of the skin lesion.</LI>
            <LI>AI analyzes the image for patterns and features.</LI>
            <LI>Receive predictions and confidence scores instantly.</LI>
          </UL>
        </FeatureCard>

        <FeatureCard>
          <h2>Features</h2>
          <UL>
            <LI>Detects 7 types of skin lesions.</LI>
            <LI>Uses ensemble learning for high accuracy.</LI>
            <LI>Simple and fast interface for ease of use.</LI>
          </UL>
        </FeatureCard>
      </FeaturesSection>

       <Container>
            <h2 style={{ color : '#52525b'}}>
                Skin disease predictor tool
            </h2>
            <Section>
                
                <ImageUploadSection>
                    <ImageUpload />
                </ImageUploadSection>
                <PredectedResultSection>
                </PredectedResultSection>
            </Section>
    
        </Container>

      <Footer>
        &copy; {new Date().getFullYear()} Skin Disease Detection. All rights reserved.
      </Footer>
    </LandingWrapper>
    )
}

export default Home