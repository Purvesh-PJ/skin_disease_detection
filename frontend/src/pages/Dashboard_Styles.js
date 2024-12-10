import styled from "styled-components";

export const Container = styled.div`
    display : flex;
    flex-direction : column;
    height: calc(100vh - 2px);
    // background-color: #f1f5f9;
`;

export const Header = styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    height: 50px;
    background-color: white;
    padding: 0 20px;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px;
`;

export const Main = styled.div`
    display: flex;
    justify-content: center;
    // align-items: center;
    margin-top : 3rem;
    height: calc(100vh - 52px);
`;

export const HeadingContainer = styled.div`
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative; 
`;

export const ProfileContainer = styled.div`
    display: flex;
    align-items: center;
    position: relative;
    cursor: pointer;
    background-color : #f1f5f9;
    border-radius : 50px;
    padding-right : 5px;
    border : 1px solid #e2e8f0;

    &:hover {
        background-color : white;
    }
`;

export const Heading = styled.h1`
    font-size: 18px;
    font-weight: 500;
    color: #475569;
    text-align: center;
    margin: 0;
`;

export const Profile = styled.div`
    display: flex;
    align-items: center;
`;

export const ProfileImage = styled.img`
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background-color: #cbd5e1;
    object-fit: cover;
    margin-right: 10px;
    // box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;
    // border : 1px solid #cbd5e1;
`;

export const Username = styled.span`
    font-size: 14px;
    font-weight: 500;
    letter-spacing : 0.05rem;
    color: #475569;
`;

export const DropdownMenu = styled.ul`
    position: absolute;
    top: 100%;
    right: 0;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    margin-top: 10px;
    list-style: none;
    padding: 4px;
    width: 100px;
    z-index: 1000;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
`;

export const DropdownItem = styled.li`
    padding: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    color: #475569;
    transition: background-color 0.2s;

    &:hover {
        background-color: #f4f4f4;
    }
`;

