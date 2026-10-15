import { FiActivity, FiClock, FiZap } from 'react-icons/fi';
import { Header } from '../../components/layout';
import { ImageUploadCard, ResultsCard } from '../../components/features/prediction';
import HistoryList from '../../components/features/history/HistoryList';
import { SAMPLE_IMAGES } from '../../constants';
import { useDashboard } from '../../hooks';

const Dashboard = () => {
  const { state, actions } = useDashboard();
  const {
    selectedImage,
    imageFile,
    activeSample,
    predictionResult,
    loading,
    error,
    activeTab,
    historyRefreshKey,
    userUpdateKey,
    hasScans,
  } = state;

  const {
    setSelectedImage,
    setImageFile,
    setActiveSample,
    setLoading,
    setError,
    setActiveTab,
    handlePredictionSuccess,
    handleUserUpdated,
    handleSelectSample,
  } = actions;

  return (
    <div key={userUpdateKey} className="flex flex-col h-screen bg-white dark:bg-[#121212] text-neutral-900 dark:text-neutral-100 overflow-hidden lg:h-screen max-lg:h-auto max-lg:overflow-y-auto">
      <Header onUserUpdated={handleUserUpdated} />

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 w-full m-0 p-0 overflow-hidden max-lg:overflow-visible">
        {/* Left Column: Image Dropzone & Bottom Sample Rail */}
        <div className="flex flex-col h-full overflow-hidden">
          <div className="flex-1 flex flex-col justify-center items-center px-6 py-4 overflow-y-auto">
            <ImageUploadCard
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              imageFile={imageFile}
              setImageFile={setImageFile}
              setPredictionResult={handlePredictionSuccess}
              activeSample={activeSample}
              setActiveSample={setActiveSample}
              loading={loading}
              setLoading={setLoading}
              setError={setError}
            />
          </div>

          {/* Bottom Sample Rail strictly inside Left Column */}
          <div className="w-full border-t border-neutral-200/80 dark:border-neutral-800/80 bg-neutral-50 dark:bg-[#0a0a0a] px-6 py-2.5 flex items-center gap-4 max-sm:px-3 max-sm:py-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-neutral-500 dark:text-neutral-400 whitespace-nowrap">
              <FiZap className="text-amber-500" size={14} />
              <span>Demo Samples:</span>
            </div>

            <div className="flex items-center gap-2.5 overflow-x-auto w-full py-1 scrollbar-thin">
              {SAMPLE_IMAGES.map((sample) => {
                const isSelected = activeSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    type="button"
                    title={`${sample.name} (${sample.code})`}
                    className={`flex items-center gap-2 px-2.5 py-1 rounded-xl border text-left shrink-0 transition-all duration-150 active:scale-[0.98] ${
                      isSelected
                        ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30 shadow-sm'
                        : 'border-neutral-200 dark:border-neutral-800 bg-white dark:bg-[#181818] hover:border-emerald-400/80 hover:bg-neutral-50/50 dark:hover:bg-neutral-900/50'
                    }`}
                  >
                    <img
                      src={sample.imagePath}
                      alt={sample.code}
                      className="w-11 h-11 rounded-md object-cover"
                    />
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-neutral-900 dark:text-neutral-100">
                        {sample.code}
                      </span>
                      <span className="text-[0.65rem] text-neutral-400 dark:text-neutral-500">
                        {sample.typeLabel || sample.type}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Analysis Results or History */}
        <div className="flex flex-col h-full border-l max-lg:border-l-0 max-lg:border-t border-neutral-200/80 dark:border-neutral-800/80 px-6 py-4 overflow-y-auto">
          {/* Tab Bar */}
          <div className="flex items-center justify-between -mx-6 -mt-4 mb-4 px-6 py-3 border-b border-neutral-200/80 dark:border-neutral-800/80 bg-white dark:bg-[#121212]">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setActiveTab('analysis')}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                  activeTab === 'analysis'
                    ? 'bg-emerald-600 border-emerald-600 text-white'
                    : 'bg-transparent border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400 hover:border-emerald-500'
                }`}
              >
                <FiActivity size={14} />
                Analysis Result
              </button>
              <button
                type="button"
                onClick={() => {
                  if (hasScans) setActiveTab('history');
                }}
                disabled={!hasScans}
                title={hasScans ? 'View diagnostic scan history' : 'Perform an image scan to see history'}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                  activeTab === 'history'
                    ? 'bg-emerald-600 border-emerald-600 text-white'
                    : hasScans
                    ? 'bg-transparent border-neutral-200 dark:border-neutral-800 text-neutral-600 dark:text-neutral-400 hover:border-emerald-500 cursor-pointer'
                    : 'bg-transparent border-transparent text-neutral-400 opacity-40 cursor-not-allowed'
                }`}
              >
                <FiClock size={14} />
                See History
              </button>
            </div>
          </div>

          <div className={`h-full flex flex-col ${activeTab === 'analysis' ? 'flex' : 'hidden'}`}>
            <ResultsCard
              predictionResult={predictionResult}
              loading={loading}
              error={error}
            />
          </div>

          <div className={`h-full flex flex-col ${activeTab === 'history' ? 'flex' : 'hidden'}`}>
            <HistoryList refreshTrigger={historyRefreshKey} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
