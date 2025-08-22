import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset { id: number; filename: string; }

type ClusteringFamily = 'graph' | 'density';

type GraphAlgorithm = 'Composantes Connexes' | 'Détection de Communautés';
type DensityAlgorithm = 'DBSCAN' | 'OPTICS' | 'DENCLUE';

type DensityMetric = 'pré-calculée';

const ClusteringPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [duplicatePairsFile, setDuplicatePairsFile] = useState<File | null>(null);

  // Clustering method and params
  const [clusteringFamily, setClusteringFamily] = useState<ClusteringFamily>('graph');
  const [graphAlgorithm, setGraphAlgorithm] = useState<GraphAlgorithm>('Composantes Connexes');
  const [edgeWeightThreshold, setEdgeWeightThreshold] = useState<number>(0.7);
  const [densityAlgorithm, setDensityAlgorithm] = useState<DensityAlgorithm>('DBSCAN');
  const [dbscanEpsilon, setDbscanEpsilon] = useState<number>(0.5);
  const [minSamples, setMinSamples] = useState<number>(2);
  const [densityMetric, setDensityMetric] = useState<DensityMetric>('pré-calculée');

  // Visualization toggles
  const [showNetworkGraph, setShowNetworkGraph] = useState<boolean>(false);
  const [showClusterDistribution, setShowClusterDistribution] = useState<boolean>(false);
  const [enableInteractiveExploration, setEnableInteractiveExploration] = useState<boolean>(false);

  // Actions/results
  const [clusteringDone, setClusteringDone] = useState(false);
  const [clusters, setClusters] = useState<any[]>([]);
  const [networkGraphData, setNetworkGraphData] = useState<any | null>(null);
  const [clusterDistributionData, setClusterDistributionData] = useState<any | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [resultsPath, setResultsPath] = useState<string | null>(null);
  const [visualizationPath, setVisualizationPath] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [loadingClustering, setLoadingClustering] = useState<boolean>(false);
  // Filename entered by user for downloaded CSV
  const [outputName, setOutputName] = useState<string>('');
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  // Helper to convert backend artifact paths (with backslashes) into accessible URLs
  const normalizePath = (p?: string): string => {
    if (!p) return '';
    // If already a full URL, optionally append the session_id
    if (p.startsWith('http')) {
      if (!sessionId) return p;
      return p.includes('?')
        ? `${p}&session_id=${encodeURIComponent(sessionId)}`
        : `${p}?session_id=${encodeURIComponent(sessionId)}`;
    }
    // Replace Windows backslashes with forward slashes and ensure leading slash removed once
    const normalized = p.replace(/\\/g, '/').replace(/^\/+/, '');
    const base = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const url = `${base}/${normalized}`;
    if (!sessionId) return url;
    return url.includes('?')
      ? `${url}&session_id=${encodeURIComponent(sessionId)}`
      : `${url}?session_id=${encodeURIComponent(sessionId)}`;
  };

  // Fetch datasets
  useEffect(() => {
    if (!token) return;
    setLoadingDatasets(true);
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets`, {
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    })
      .then(async res => {
        if (!res.ok) throw new Error('Failed to load datasets');
        const data = await res.json();
        setDatasets(data);
        setDatasetsError(null);
      })
      .catch(() => {
        setDatasets([]);
        setDatasetsError('Failed to load datasets');
      })
      .finally(() => setLoadingDatasets(false));
  }, [token]);

  // Real clustering action
  const handleClustering = async () => {
    if (!token) return;
    if (!selectedDataset) { setErrorMsg('Please select a dataset'); return; }
    if (!duplicatePairsFile) { setErrorMsg('Please upload a classification results file'); return; }

    try {
      setLoadingClustering(true);
      setErrorMsg(null);
      setClusteringDone(false);
      setSessionStepId(null);
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

      // 1. Upload the classification-results CSV/JSON
      const form = new FormData();
      form.append('file', duplicatePairsFile);
      const upRes = await fetch(`${API_URL}/api/v1/artifacts/upload${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: form
      });
      if (!upRes.ok) throw new Error('File upload failed');
      const upData = await upRes.json();
      const classificationPath = upData.artifact_path;

      // 2. Resolve method string
      const methodKey = `${clusteringFamily}_${clusteringFamily === 'graph' ? graphAlgorithm : densityAlgorithm}`;
      const methodMap: Record<string, string> = {
        'graph_Composantes Connexes': 'graph_connected_components',
        'graph_Détection de Communautés': 'graph_community_detection',
        'density_DBSCAN': 'dbscan',
        'density_OPTICS': 'optics',
        'density_DENCLUE': 'denclue'
      };
      const method = methodMap[methodKey];

      // 3. Params
      const params: any = { output_name: outputName };
      if (clusteringFamily === 'graph') {
        params.confidence_threshold = edgeWeightThreshold;
      } else {
        if (densityAlgorithm === 'DBSCAN') {
          params.eps = dbscanEpsilon;
          params.min_samples = minSamples;
        } else if (densityAlgorithm === 'OPTICS') {
          params.min_samples = minSamples;
          params.max_eps = dbscanEpsilon;
        }
      }

      // 4. POST request
      const payload = {
        dataset_id: Number(selectedDataset),
        classification_results_path: classificationPath,
        method,
        params
      };

      const res = await fetch(`${API_URL}/api/v1/deduplication/pipeline/clustering${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Clustering error: ${errText || res.status}`);
      }
      const data = await res.json();
      if (typeof data.session_step_id !== 'undefined') {
        setSessionStepId(data.session_step_id || null);
      }
      if (data.status === 'error') {
        throw new Error(data.message || 'Clustering failed');
      }

      setSummary(data.summary);
      setClusters(data.preview || []);
      setResultsPath(data.clustering_results_path || null);
      setVisualizationPath(data.visualization_path || null);
      if (showNetworkGraph && data.visualization_path) {
        setNetworkGraphData({ path: data.visualization_path });
      } else {
        setNetworkGraphData(null);
      }
      setClusteringDone(true);
    } catch (err: any) {
      setErrorMsg(err.message || 'Unknown error');
    } finally {
      setLoadingClustering(false);
    }
  };

  // UI
  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-center">Clustering des Doublons</h1>
      {sessionId && (
        <div className="mb-2 text-xs text-gray-500 text-center">
          Active session: <span className="font-mono">{sessionId}</span>
          {sessionStepId && (
            <>
              {' '}• Dernière étape: <span className="font-mono">{sessionStepId}</span>
            </>
          )}
        </div>
      )}
      {/* Input Data */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Input Data</h2>
        <label className="block mb-2 font-medium">Use original dataset</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={loadingDatasets || datasets.length === 0}
        >
          {loadingDatasets && <option>Loading...</option>}
          {!loadingDatasets && datasets.length === 0 && <option>No dataset available</option>}
          {!loadingDatasets && datasets.length > 0 && (
            <>
              <option value="">-- Select a dataset --</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.filename}</option>
              ))}
            </>
          )}
        </select>
        {datasetsError && <div className="text-red-600 text-sm mb-2">{datasetsError}</div>}
        <label className="block mb-2 font-medium">Upload classified duplicate pairs</label>
        <input
          type="file"
          accept=".csv,.json"
          className="block w-full mb-2"
          onChange={e => setDuplicatePairsFile(e.target.files?.[0] || null)}
        />
      </div>

      {/* Clustering Method */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Clustering Method</h2>
        <div className="flex gap-4 mb-4">
          <button
            className={`px-4 py-2 rounded font-semibold border ${clusteringFamily === 'graph' ? 'bg-blue-500 text-white' : 'bg-white text-blue-500 border-blue-500'}`}
            onClick={() => setClusteringFamily('graph')}
          >
            Graph-based
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold border ${clusteringFamily === 'density' ? 'bg-blue-500 text-white' : 'bg-white text-blue-500 border-blue-500'}`}
            onClick={() => setClusteringFamily('density')}
          >
            Density-based
          </button>
        </div>
        {clusteringFamily === 'graph' && (
          <div className="flex flex-col gap-2 mb-2">
            <label className="font-medium">Algorithm</label>
            <select
              className="p-2 border rounded"
              value={graphAlgorithm}
              onChange={e => setGraphAlgorithm(e.target.value as GraphAlgorithm)}
            >
              <option value="Composantes Connexes">Connected Components</option>
              <option value="Détection de Communautés">Community Detection</option>
            </select>
            <label className="font-medium">Edge Weight Threshold (Similarity)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={edgeWeightThreshold}
              onChange={e => setEdgeWeightThreshold(parseFloat(e.target.value))}
            />
            <span>{edgeWeightThreshold}</span>
          </div>
        )}
        {clusteringFamily === 'density' && (
          <div className="flex flex-col gap-2 mb-2">
            <label className="font-medium">Algorithm</label>
            <select
              className="p-2 border rounded"
              value={densityAlgorithm}
              onChange={e => setDensityAlgorithm(e.target.value as DensityAlgorithm)}
            >
              <option value="DBSCAN">DBSCAN</option>
              <option value="OPTICS">OPTICS</option>
              <option value="DENCLUE">DENCLUE</option>
            </select>
            {densityAlgorithm === 'DBSCAN' && (
              <>
                <label className="font-medium">Epsilon (DBSCAN)</label>
                <input
                  type="number"
                  step="0.01"
                  className="p-2 border rounded"
                  value={dbscanEpsilon}
                  min={0}
                  onChange={e => setDbscanEpsilon(Number(e.target.value))}
                />
              </>
            )}
            <label className="font-medium">Minimum Samples (Min Samples)</label>
            <input
              type="number"
              className="p-2 border rounded"
              value={minSamples}
              min={1}
              onChange={e => setMinSamples(Number(e.target.value))}
            />
            <label className="font-medium">Distance Metric</label>
            <select
              className="p-2 border rounded"
              value={densityMetric}
              onChange={e => setDensityMetric(e.target.value as DensityMetric)}
            >
              <option value="pré-calculée">pre-computed</option>
            </select>
          </div>
        )}
      </div>

      {/* Visualization Options */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Visualization Options</h2>
        <div className="flex flex-col gap-2">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showNetworkGraph}
              onChange={e => setShowNetworkGraph(e.target.checked)}
              className="mr-2"
            />
            Show Network Graph
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={showClusterDistribution}
              onChange={e => setShowClusterDistribution(e.target.checked)}
              className="mr-2"
            />
            Show Cluster Distribution
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={enableInteractiveExploration}
              onChange={e => setEnableInteractiveExploration(e.target.checked)}
              className="mr-2"
            />
            Enable Interactive Exploration
          </label>
        </div>
      </div>

      {/* Actions */}
      <div className="mb-6 flex flex-wrap gap-4">
        <button
          className="px-4 py-2 rounded text-white bg-blue-500 hover:bg-blue-600 font-semibold disabled:opacity-60"
          onClick={handleClustering}
          disabled={loadingClustering}
        >
          {loadingClustering ? 'Running...' : 'Run Clustering'}
        </button>
      </div>

      {errorMsg && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
          {errorMsg}
        </div>
      )}
      {clusteringDone && summary && (
        <div className="my-4 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Clustering Summary</h2>
          {/* filename + download */}
          {resultsPath && (
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-4">
              <input
                type="text"
                className="p-2 border rounded w-full sm:flex-grow"
                placeholder="Custom filename (optional)"
                value={outputName}
                onChange={e => setOutputName(e.target.value)}
              />
              <a
                href={normalizePath(resultsPath)}
                download={`${outputName || 'clustering_results'}.csv`}
                className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-800 whitespace-nowrap text-center"
              >
                Download Cluster Assignments
              </a>
            </div>
          )}
          <ul className="list-disc list-inside text-sm space-y-1">
            <li><span className="font-medium">Method:</span> {summary.method}</li>
            {summary.params?.confidence_threshold && (
              <li><span className="font-medium">Confidence Threshold:</span> {summary.params.confidence_threshold}</li>
            )}
            <li><span className="font-medium">Total Clusters:</span> {summary.total_clusters}</li>
            <li><span className="font-medium">Total Records in Clusters:</span> {summary.total_records_in_clusters}</li>
          </ul>
        </div>
      )}
      {clusteringDone && visualizationPath && (
        <div className="my-4 p-3 bg-green-50 border border-green-200 text-green-800 rounded">
          <h3 className="font-semibold mb-2">Cluster Visualization</h3>
          <img src={normalizePath(visualizationPath)} alt="Cluster visualization" className="mx-auto max-w-full" style={{ maxHeight: '450px', objectFit: 'contain' }} />
        </div>
      )}

      {/* Visualizations */}
      {clusteringDone && showNetworkGraph && networkGraphData && (
        <div className="my-4 p-3 bg-purple-50 border border-purple-200 text-purple-800 rounded">
          <h3 className="font-semibold mb-2">Network Graph (example)</h3>
          <pre>{JSON.stringify(networkGraphData, null, 2)}</pre>
        </div>
      )}
      {clusteringDone && showClusterDistribution && clusterDistributionData && (
        <div className="my-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded">
          <h3 className="font-semibold mb-2">Cluster Distribution (example)</h3>
          {/* <pre>{JSON.stringify(clusterDistributionData, null, 2)}</pre> */}
        </div>
      )}
      {/* Result Table */}
      {clusteringDone && (
        <div className="my-4 p-3 bg-blue-50 border border-blue-200 text-blue-800 rounded">
          <h3 className="font-semibold mb-2">Example Clusters:</h3>
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr>
                <th className="px-2 py-1">Cluster ID</th>
                <th className="px-2 py-1">Record IDs</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((cl, idx) => {
                const idsCandidate = cl.record_ids ?? (cl.records ? cl.records.map((r: any) => r.record_id) : []);
                const ids: number[] = Array.isArray(idsCandidate) ? idsCandidate : [];
                return (
                  <tr key={idx}>
                    <td className="px-2 py-1">{cl.cluster_id ?? idx}</td>
                    <td className="px-2 py-1">{ids.length ? ids.join(', ') : ''}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default ClusteringPage;
