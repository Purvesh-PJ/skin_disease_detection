import { useState } from "react";
import { predictSkinDiseaseFromImageApi } from "../services/predictSkinDiseaseFromImageApi";


const usePostImageForPrediction  = () => {

    const [ error, setError ] = useState(null);
    const [ loading, setLoading ] = useState(false);

    const postImageToPredict = async (ImageData) => {
        setLoading(true);
        try {
            const response = await predictSkinDiseaseFromImageApi(ImageData);
            return response;
        } 
        catch(error) {
            console.log(error);
            setError(error);  
        }
        finally {
            setLoading(false);
        }
    };

    return {
        postImageToPredict,
        error,
        loading
    }
};

export default usePostImageForPrediction;