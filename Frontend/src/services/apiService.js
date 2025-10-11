import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor to add auth token
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  response => response.data,
  error => {
    // Handle special cases (like 409 Conflict) by preserving full error data
    if (error.response?.status === 409) {
      const conflictError = new Error(error.response.data.message || 'Conflict');
      conflictError.statusCode = 409;
      conflictError.dependencies = error.response.data.dependencies;
      conflictError.response = error.response;
      return Promise.reject(conflictError);
    }
    
    // Handle authentication errors
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      // Only redirect to login if not already on login/register pages
      if (!window.location.pathname.includes('/login') && !window.location.pathname.includes('/register')) {
        window.location.href = '/login';
      }
    }
    
    // Extract error message from various possible formats
    const message = error.response?.data?.detail 
      || error.response?.data?.message 
      || error.message 
      || 'An error occurred';
    
    const apiError = new Error(message);
    apiError.response = error.response;
    apiError.statusCode = error.response?.status;
    return Promise.reject(apiError);
  }
);

// Authentication API
export const authAPI = {
  login: (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    return api.post('/api/v1/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  register: (email, password) => 
    api.post('/api/v1/auth/register', { email, password }),
  getCurrentUser: () => api.get('/api/v1/auth/me'),
  logout: () => api.post('/api/v1/auth/logout')
};

export const datasetAPI = {
  upload: (file, name = null, description = null) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name || file.name.replace(/\.[^/.]+$/, ""));
    if (description) {
      formData.append('description', description);
    }
    return api.post('/api/v1/datasets/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  list: () => api.get('/api/v1/datasets/').then(response => response.datasets || response),
  get: (id) => api.get(`/api/v1/datasets/${id}`),
  preview: (id, rows = 10) => 
    api.get(`/api/v1/datasets/${id}/preview`, { params: { rows } }),
  update: (id, data) => api.put(`/api/v1/datasets/${id}`, data),
  delete: (id) => api.delete(`/api/v1/datasets/${id}`)
};

// EDA API
export const edaAPI = {
  getSummary: (datasetId) => api.get(`/api/v1/datasets/${datasetId}/eda`),
  generatePlot: (datasetId, plotConfig) => 
    api.post(`/api/v1/datasets/${datasetId}/plot`, plotConfig)
};

export const experimentAPI = {
  create: (data) => api.post('/api/v1/experiments/', data),
  list: (page = 1, pageSize = 20) => api.get('/api/v1/experiments/', { params: { page, page_size: pageSize } }),
  get: (id) => api.get(`/api/v1/experiments/${id}`),
  update: (id, data) => api.put(`/api/v1/experiments/${id}`, data),
  delete: (id) => api.delete(`/api/v1/experiments/${id}`),
  duplicate: (id) => api.post(`/api/v1/experiments/${id}/duplicate`)
};

// New pipeline-based preprocessing API
export const preprocessAPI = {
  // Pipeline step management
  addStep: (experimentId, step) => api.post(`/api/experiments/${experimentId}/pipeline/steps`, step),
  updateStep: (experimentId, stepId, config) => api.put(`/api/experiments/${experimentId}/pipeline/steps/${stepId}`, config),
  deleteStep: (experimentId, stepId) => api.delete(`/api/experiments/${experimentId}/pipeline/steps/${stepId}`),
  
  // Execute the pipeline (returns jobId for async processing)
  apply: (experimentId) => api.post(`/api/experiments/${experimentId}/pipeline/apply`),
  
  // Get EDA results
  getEDA: (experimentId) => api.get(`/api/experiments/${experimentId}/eda`)
};

// Training API - Run-based architecture
export const trainingAPI = {
  // NEW RUN-BASED ENDPOINTS
  // Start training run (config-based, trains all models)
  startRun: (experimentId) => 
    api.post(`/api/v1/training/experiments/${experimentId}/runs`),
  
  // List all runs for experiment
  listRuns: (experimentId, page = 1, pageSize = 20) =>
    api.get(`/api/v1/training/experiments/${experimentId}/runs`, { 
      params: { page, page_size: pageSize } 
    }),
  
  // Get full run details with results
  getRunDetails: (runId) => api.get(`/api/v1/training/runs/${runId}`),
  
  // Get run status (for polling)
  getRunStatus: (runId) => api.get(`/api/v1/training/runs/${runId}/status`),
  
  // LEGACY JOB-BASED ENDPOINTS (for backward compatibility)
  // Start training for multiple models (models are read from experiment config on backend)
  start: (experimentId) => 
    api.post(`/api/v1/training/experiments/${experimentId}/train`, {}),
  
  // Get individual job status
  getJobStatus: (jobId) => api.get(`/api/v1/training/jobs/${jobId}`),
  
  // List all jobs for experiment  
  listJobs: (experimentId, statusFilter = null) => {
    const params = statusFilter ? { status_filter: statusFilter } : {};
    return api.get(`/api/v1/training/experiments/${experimentId}/jobs`, { params });
  },
  
  // Get experiment training status overview
  getExperimentStatus: (experimentId) => 
    api.get(`/api/v1/training/experiments/${experimentId}/status`),
  
  // Get training result by result ID
  getResult: (resultId) => api.get(`/api/v1/training/results/${resultId}`),
  
  // Get training result by job ID
  getResultByJob: (jobId) => api.get(`/api/v1/training/results/job/${jobId}`)
};

export const predictionAPI = {
  // Make single prediction with trained model
  single: (experimentId, features) => 
    api.post(`/api/v1/predictions/experiments/${experimentId}/predict/single`, { features }),
  
  // Make batch predictions from CSV file
  batch: (experimentId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/api/v1/predictions/experiments/${experimentId}/predict/batch`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  
  // Get prediction history for an experiment
  getHistory: (experimentId) => api.get(`/api/v1/predictions/experiments/${experimentId}/history`),
  
  // Download prediction results with authentication
  download: async (predictionId) => {
    // Make direct axios request to bypass response interceptor that extracts .data
    const response = await axios.get(
      `${API_BASE_URL}/api/v1/predictions/download/${predictionId}`,
      {
        responseType: 'blob',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('auth_token')}`
        }
      }
    );
    
    // response.data is the blob from axios
    const url = window.URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `predictions_${predictionId}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return response;
  },
  
  // Legacy URL method (no auth)
  downloadUrl: (predictionId) => `${API_BASE_URL}/api/v1/predictions/download/${predictionId}`
};

export const modelAPI = {
  get: (modelId) => api.get(`/api/models/${modelId}`),
  export: (modelId) => api.get(`/api/models/${modelId}/export`, {
    responseType: 'blob'
  })
};

// ML Models list API
export const modelsAPI = {
  getAll: () => api.get('/api/v1/models'),
  getByTask: (taskType) => api.get(`/api/v1/models/${taskType}`)
};

// Global job status API for tracking async operations
export const jobAPI = {
  getStatus: (jobId) => api.get(`/api/jobs/${jobId}/status`)
};

// Convenience methods for backwards compatibility
const apiService = {
  // Training methods for TrainingContext
  startTraining: (experimentId) => trainingAPI.start(experimentId),
  getTrainingJob: (jobId) => trainingAPI.getJobStatus(jobId),
  getExperimentJobs: (experimentId) => trainingAPI.listJobs(experimentId),
  getExperimentTrainingStatus: (experimentId) => trainingAPI.getExperimentStatus(experimentId),
  getTrainingResult: (resultId) => trainingAPI.getResult(resultId),
  getTrainingResultByJob: (jobId) => trainingAPI.getResultByJob(jobId),
  
  // Re-export other APIs
  auth: authAPI,
  dataset: datasetAPI,
  eda: edaAPI,
  experiment: experimentAPI,
  preprocess: preprocessAPI,
  training: trainingAPI,
  prediction: predictionAPI,
  model: modelAPI,
  models: modelsAPI,
  job: jobAPI
};

export default apiService;
