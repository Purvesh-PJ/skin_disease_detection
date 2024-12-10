import axios from 'axios';
import { BACKEND_BASE_URL } from './config';


export const predictApi = async (imageData) => {
    const response = await axios.post(`${BACKEND_BASE_URL}/predict`, imageData, { 
        headers: {
        'Content-Type' : 'multipart/form-data'
    }});
    return response;
};