import styled from "styled-components";

export const Container = styled.div`
    height: calc(100vh - 2px);
    background-color : #f1f5f9;
    // border : 1px solid gray;
`;

export const Header = styled.div`
    display : flex;
    flex-direction : row;
    // justify-content : space-around;
    align-items : center;
    height : 50px;
    background-color : white;
    // border : 1px solid gray;
`;

export const HeadingContainer = styled.div`
    width : 95%;
    min-width : 100px;
    margin-left : auto;
    margin-right : auto;
    // border : 1px solid gray;
`;

export const ProfileContainer = styled.div`
    margin-right : auto;
    // border : 1px solid gray;
`;

export const Heading = styled.h1`
    font-size : 14px;
    font-weight : 500;
    color : #475569;
    text-align : center;
`;

export const Profile = styled.div`
    width : 40px;
    height : 40px;
    border-radius : 50%;
    background-color : #cbd5e1;
`;

export const Main = styled.div`
    display : flex;
    justify-content : center;
    align-items : center;
    height: calc(100vh - 52px);
    //  border : 1px solid gray;
`;