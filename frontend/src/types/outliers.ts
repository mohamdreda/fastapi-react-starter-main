export interface Dataset {
  id: string;
  filename: string;
  createdAt: string;
}

export interface OutlierAnalysisRequest {
  dataset: string;
  featureExtraction: 'Autoencoder' | 'PCA' | 'ISOMAP';
  clustering: 'DENCLUE' | 'DBSCAN' | 'OPTICS';
  outlierDetection: 'IF' | 'LOF' | 'OCSVM';
}

export interface OutlierAnalysisResponse {
  success: boolean;
  message: string;
  results?: {
    outliers: number[];
    scores?: number[];
    parameters: any;
  };
}
