import predictionService from '../../services/api/prediction.service';
import axiosInstance from '../../services/api/axios';

jest.mock('../../services/api/axios', () => ({
  post: jest.fn(),
}));

describe('predictionService Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('calls /predict with FormData and multipart header', async () => {
    const mockResponse = {
      data: {
        predicted_disease: 'nv',
        confidence: '91',
        filename: 'test.jpg',
      },
    };
    axiosInstance.post.mockResolvedValueOnce(mockResponse);

    const formData = new FormData();
    formData.append('image', new Blob(['fake image content']), 'test.jpg');

    const result = await predictionService.predict(formData);

    expect(axiosInstance.post).toHaveBeenCalledTimes(1);
    expect(axiosInstance.post).toHaveBeenCalledWith(
      '/predict',
      formData,
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
    );
    expect(result.data.predicted_disease).toBe('nv');
  });

  test('propagates rejection on API error', async () => {
    const networkError = new Error('Network Error');
    axiosInstance.post.mockRejectedValueOnce(networkError);

    const formData = new FormData();

    await expect(predictionService.predict(formData)).rejects.toThrow('Network Error');
  });
});

