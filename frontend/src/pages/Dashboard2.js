import { Container, Header, Main, Heading, Profile, HeadingContainer, ProfileContainer } from "./Dashboard2_Styles";
import DiseasePredictorTool from "../components/DiseasePredictorTool";


const Dashboard2 = () => {

    return (
        <Container>
            <Header>
                <HeadingContainer>
                    <Heading>Skin Disease Predictor</Heading>
                </HeadingContainer>
                <ProfileContainer>
                    <Profile></Profile>
                </ProfileContainer>
            </Header>
            <Main>
                <DiseasePredictorTool/>
            </Main>
        </Container>
    )
}

export default Dashboard2;