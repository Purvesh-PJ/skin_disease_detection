import { useState } from "react";
import { predictApi } from '../services/predictApi'


const useSkinDiseasePrediction  = () => {

    const [ error, setError ] = useState(null);
    const [ loading, setLoading ] = useState(false);

    const postImageToPredict = async (ImageData) => {
        setLoading(true);
        try {
            const response = await predictApi(ImageData);
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

export default useSkinDiseasePrediction;