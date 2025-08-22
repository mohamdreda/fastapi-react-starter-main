import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Fetch columns for a dataset by its ID.
 * Returns an array of column names (string[])
 */
export async function getDatasetColumns(datasetId: string, token: string): Promise<string[]> {
  if (!datasetId) return [];
  try {
    const res = await axios.get(`${API_BASE_URL}/api/v1/datasets/${datasetId}/columns`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (res.status === 200 && Array.isArray(res.data)) {
      return res.data;
    }
    // Some APIs return { columns: [...] }
    if (res.status === 200 && Array.isArray(res.data.columns)) {
      return res.data.columns;
    }
    return [];
  } catch (err) {
    // Optionally: log or show error
    return [];
  }
}
