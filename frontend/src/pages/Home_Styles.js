import styled from 'styled-components';

export const LandingWrapper = styled.div`
  font-family: 'Roboto', sans-serif;
  background: linear-gradient(180deg, #f9f9f9, #e8f5e9);
  color: #333;
  padding: 0;
  margin: 0;
`;

export const HeroSection = styled.section`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  background: #4caf50;
  color: white;
  padding: 50px 20px;

  h1 {
    font-size: 3rem;
    margin: 10px 0;
  }

  p {
    font-size: 1.2rem;
    margin: 10px 0 20px;
    line-height: 1.6;
  }

  a {
    text-decoration: none;
    background-color: #ffffff;
    color: #4caf50;
    padding: 12px 24px;
    font-size: 1rem;
    font-weight: bold;
    border-radius: 5px;
    transition: background-color 0.3s ease, color 0.3s ease;

    &:hover {
      background-color: #45a049;
      color: white;
    }
  }
`;

export const FeaturesSection = styled.section`
  max-width: 1200px;
  margin: 40px auto;
  padding: 20px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
`;

export const FeatureCard = styled.div`
  background: white;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  padding: 20px;
  text-align: left;
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
  }

  h2 {
    font-size: 1.5rem;
    color: #4caf50;
    margin-bottom: 10px;
  }

  p {
    font-size: 1rem;
    line-height: 1.6;
    color: #555;
  }

  ul {
    padding-left: 20px;
    color: #555;
  }
`;

export const Footer = styled.footer`
  background: #4caf50;
  color: white;
  text-align: center;
  padding: 10px;
  font-size: 0.9rem;
`;

export const UL = styled.ul`
`;

export const LI = styled.li`
    margin-bottom : 8px;
    list-style-type : disc;
`;

export const Container = styled.div`
    display : flex;
    flex-direction : column;
    justify-content : center;
    align-items : center; 
    margin-bottom : 2rem;
`;

export const Section = styled.section`
    display : flex;
    flex-direction : row;
    gap : 1px;
    width : 100%;
    max-width : 1250px;
    height : 800px;
    border-radius : 10px;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 6px 24px 0px, rgba(0, 0, 0, 0.08) 0px 0px 0px 1px;
    background-color : #e5e7eb;
    // border : 1px solid lightgray;
`;

export const ImageUploadSection = styled.div`
    display : flex;
    justify-content : center;
    width : 50%;
    background-color : white;
    border-top-left-radius : 10px;
    border-bottom-left-radius : 10px;
    // border : 1px solid lightgray;
`;

export const PredectedResultSection = styled.div`
    width : 50%;
    background-color : white;
    border-top-right-radius : 10px;
    border-bottom-right-radius : 10px;
    box-sizing : border-box;
    // border : 1px solid lightgray;
`;