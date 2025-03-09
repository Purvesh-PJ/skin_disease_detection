import React, { useState } from "react";
import { Container, Header, Main, Heading, HeadingContainer, ProfileContainer, ProfileImage, Username, DropdownMenu, DropdownItem } from "./Dashboard_Styles";
import DiseasePredictorTool from "../components/DiseasePredictorTool";
import Default_Profile from "../resources/images/default_profile.jpg";
import { logout } from "../services/authApi";


const Dashboard = () => {

  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const User = JSON.parse(localStorage.getItem('user'));

  const toggleDropdown = () => {
    setDropdownOpen((prev) => !prev);
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <Container>

      <Header>
        <div style={{ width: '150px' }}></div>
        
        <HeadingContainer>
          <Heading>Skin Disease Predictor</Heading>
        </HeadingContainer>

        <ProfileContainer onClick={toggleDropdown}>
          <ProfileImage src={Default_Profile} />
          <Username>{ User ? User.username : "User" }</Username>
          <div style={{ width: '5px' }}></div>
  
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

export default Dashboard;
