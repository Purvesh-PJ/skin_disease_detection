import React, { useState } from "react";
import { Container, Header, Main, Heading, Profile, HeadingContainer, ProfileContainer, ProfileImage, Username, DropdownMenu, DropdownItem } from "./Dashboard2_Styles";
import DiseasePredictorTool from "../components/DiseasePredictorTool";
import Default_Profile from "../resources/images/default_profile.jpg";

const Dashboard2 = () => {

  const [isDropdownOpen, setDropdownOpen] = useState(false);

  const toggleDropdown = () => {
    setDropdownOpen((prev) => !prev);
  };

  const handleLogout = () => {
    console.log("Logging out...");
  };

  return (
    <Container>
      <Header>
        <HeadingContainer>
          <Heading>Skin disease predictor</Heading>
        </HeadingContainer>

        <ProfileContainer onClick={toggleDropdown}>

            <ProfileImage
              src={Default_Profile} // Replace with user profile image URL
              alt="User"
            />
            <Username>John Doe</Username>
  
           {isDropdownOpen && (
                <DropdownMenu>
                    <DropdownItem onClick={handleLogout}>
                        Logout
                    </DropdownItem>
                </DropdownMenu>
            )}

        </ProfileContainer>
      </Header>
      <Main>
        <DiseasePredictorTool />
      </Main>
    </Container>
  );
};

export default Dashboard2;
