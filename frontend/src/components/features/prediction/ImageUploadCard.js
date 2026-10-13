import React from 'react';
import { FiUploadCloud, FiCheck } from 'react-icons/fi';
import { Spinner, Button } from '../../ui';
import { usePrediction } from '../../../hooks';

const ImageUploadCard = ({
  selectedImage,
  setSelectedImage,
  imageFile,
  setImageFile,
  setPredictionResult,
  activeSample,
  setActiveSample,
  loading,
  setLoading,
  setError
}) => {
  const { predict } = usePrediction();
  const [coldStartNotice, setColdStartNotice] = React.useState(false);

  React.useEffect(() => {
    let timer;
    if (loading) {
      timer = setTimeout(() => {
        setColdStartNotice(true);
      }, 3500);
    } else {
      setColdStartNotice(false);
    }
    return () => clearTimeout(timer);
  }, [loading]);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      if (setActiveSample) setActiveSample(null);
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
      if (setActiveSample) setActiveSample(null);
      setPredictionResult(null);
      setError(null);
    }
  };

  const handleUploadClick = async () => {
    if (!imageFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
      const response = await predict(formData);
      if (response?.data) {
        setPredictionResult(response.data);
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImageFile(null);
    if (setActiveSample) setActiveSample(null);
    setPredictionResult(null);
    setError(null);
    setColdStartNotice(false);
  };

  return (
    <div className="w-full max-w-[560px] flex flex-col items-center justify-center gap-3">
      {/* DropZone */}
      <div
        onClick={() => document.getElementById('file-input').click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className={`w-full h-[330px] flex flex-col items-center justify-center border-2 border-dashed rounded-2xl p-4 cursor-pointer transition-all relative overflow-hidden select-none ${
          selectedImage
            ? 'border-emerald-500 bg-emerald-500/[0.08] dark:bg-emerald-500/[0.08]'
            : 'border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-[#0a0a0a] hover:border-emerald-500 hover:bg-emerald-500/[0.04]'
        }`}
      >
        {selectedImage ? (
          <>
            {activeSample && (
              <div className="absolute top-2.5 left-2.5 bg-black/80 backdrop-blur-md text-white px-2.5 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 z-10 border border-white/15">
                <FiCheck size={12} className="text-emerald-400" />
                <span>{activeSample.name} ({activeSample.code})</span>
              </div>
            )}
            <img
              src={selectedImage}
              alt="Lesion Preview"
              className="w-full h-full object-contain rounded-xl"
            />
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 text-neutral-500 dark:text-neutral-400 text-center">
            <FiUploadCloud size={40} className="text-emerald-500" />
            <p className="text-[0.95rem] font-semibold text-neutral-900 dark:text-neutral-100 m-0">
              <span className="text-emerald-600 dark:text-emerald-400">Click to upload</span> or drag lesion image
            </p>
            <span className="text-xs text-neutral-400 dark:text-neutral-500">
              PNG, JPG, or JPEG
            </span>
          </div>
        )}
      </div>

      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        disabled={loading}
        className="hidden"
      />

      {/* Buttons */}
      <div className="flex gap-2.5 w-full">
        <Button
          onClick={handleUploadClick}
          disabled={loading || !imageFile}
          fullWidth
          size="lg"
        >
          {loading ? (
            <div className="flex items-center gap-2 justify-center">
              <Spinner size="sm" color="white" />
              <span>{coldStartNotice ? 'Warming up AI Ensemble...' : 'Analyzing Lesion...'}</span>
            </div>
          ) : (
            'Run Prediction'
          )}
        </Button>
        {selectedImage && (
          <Button
            variant="secondary"
            onClick={handleClear}
            disabled={loading}
            size="lg"
          >
            Clear
          </Button>
        )}
      </div>

      {coldStartNotice && loading && (
        <p className="text-xs text-neutral-400 dark:text-neutral-500 text-center m-0 mt-1 leading-snug">
          ⚡ Initializing 3-backbone neural network ensemble from model registry. Subsequent scans are instantaneous.
        </p>
      )}
    </div>
  );
};

export default ImageUploadCard;
