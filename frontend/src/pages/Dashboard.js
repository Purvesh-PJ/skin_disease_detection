import { useState, useRef, useEffect } from "react";
import {
  Container,
  Header,
  HeaderLeft,
  HeaderCenter,
  HeaderRight,
  Main,
  Heading,
  LeftPanel,
  RightPanel,
  PanelHeader,
  PanelTitle,
  PanelContent,
  ProfileContainer,
  ProfileImage,
  DropdownMenu,
  DropdownItem,
  DropdownDivider,
  Logo,
  LogoText
} from "./Dashboard_Styles";
import Default_Profile from "../resources/images/default_profile.jpg";
import { logout } from "../services/authApi";
import { ThemeToggle } from "../components/ui";
import { FiLogOut, FiUser, FiSettings, FiUploadCloud, FiActivity } from 'react-icons/fi';
import styled from "styled-components";
import ImageUploadCard from "../components/ImageUploadCard";
import ResultsCard from "../components/ResultsCard";

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary[500]}, ${({ theme }) => theme.colors.primary[700]});
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: 0.9rem;
`;

const Dashboard = () => {
  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const dropdownRef = useRef(null);
  const User = JSON.parse(localStorage.getItem('user'));

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleDropdown = () => {
    setDropdownOpen((prev) => !prev);
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <Container>
      <Header>
        <HeaderLeft>
          <Logo>
            <LogoIcon>SP</LogoIcon>
            <LogoText>Skin Disease Predictor</LogoText>
          </Logo>
        </HeaderLeft>

        <HeaderCenter>
        </HeaderCenter>

        <HeaderRight>
          <ThemeToggle />
          
          <ProfileContainer ref={dropdownRef} onClick={toggleDropdown}>
            <ProfileImage src={Default_Profile} alt="Profile" />

            {isDropdownOpen && (
              <DropdownMenu>
                <DropdownItem>
                  <FiUser size={14} />
                  {User ? User.username : "Profile"}
                </DropdownItem>
                <DropdownItem>
                  <FiSettings size={14} />
                  Settings
                </DropdownItem>
                <DropdownDivider />
                <DropdownItem className="danger" onClick={handleLogout}>
                  <FiLogOut size={14} />
                  Logout
                </DropdownItem>
              </DropdownMenu>
            )}
          </ProfileContainer>
        </HeaderRight>
      </Header>

      <Main>
        <LeftPanel>
          <PanelHeader>
            <PanelTitle>
              <FiUploadCloud size={18} />
              <h3>Upload Image</h3>
            </PanelTitle>
          </PanelHeader>
          <PanelContent>
            <ImageUploadCard
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              imageFile={imageFile}
              setImageFile={setImageFile}
              setPredictionResult={setPredictionResult}
              loading={loading}
              setLoading={setLoading}
              setError={setError}
            />
          </PanelContent>
        </LeftPanel>

        <RightPanel>
          <PanelHeader>
            <PanelTitle>
              <FiActivity size={18} />
              <h3>Analysis Results</h3>
            </PanelTitle>
          </PanelHeader>
          <PanelContent>
            <ResultsCard
              predictionResult={predictionResult}
              loading={loading}
              error={error}
            />
          </PanelContent>
        </RightPanel>
      </Main>
    </Container>
  );
};

export default Dashboard;
