import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const fetchPredictions = (data) => {
  return axios.post(`${API_URL}/predict`, data);
};