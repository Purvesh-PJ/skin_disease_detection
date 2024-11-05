import styled from 'styled-components';

export const Container = styled.div`
    display : flex;
    flex-direction : column;
    justify-content : center;
    align-items : center; 
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