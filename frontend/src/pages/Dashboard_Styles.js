import styled from "styled-components";

export const Container = styled.div`
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    max-height: 100vh;
    overflow: hidden;
`;

export const Header = styled.div`
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    height: 60px;
    background-color: white;
    padding: 0 20px;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 0px 0px 1px;
    position: relative;
    
    @media (max-width: 768px) {
        padding: 0 10px;
    }
`;

export const Main = styled.div`
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 20px;
    flex: 1;
    overflow-y: auto;
    
    @media (max-width: 768px) {
        padding: 10px;
    }
`;

export const HeadingContainer = styled.div`
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    pointer-events: none;
    z-index: 0;
`;

export const ProfileContainer = styled.div`
    display: flex;
    align-items: center;
    position: relative;
    cursor: pointer;
    background-color: #f1f5f9;
    border-radius: 50px;
    padding: 5px 5px 5px 5px;
    border: 1px solid #e2e8f0;
    z-index: 1;
    width: 150px;
    justify-content: flex-start;
    transition: background-color 0.2s ease;

    &:hover {
        background-color: white;
    }
    
    @media (max-width: 480px) {
        width: auto;
        padding: 5px;
        justify-content: center;
    }
`;

export const Heading = styled.h1`
    font-size: 18px;
    font-weight: 500;
    color: #475569;
    text-align: center;
    margin: 0;
    pointer-events: auto;
    
    @media (max-width: 768px) {
        font-size: 16px;
    }
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
    margin-right: 8px;
`;

export const Username = styled.span`
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.05rem;
    color: #475569;
    flex: 1;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    
    @media (max-width: 480px) {
        display: none;
    }
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

