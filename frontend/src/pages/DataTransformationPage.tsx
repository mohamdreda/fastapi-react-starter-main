import React, { useEffect, useState } from 'react';
import api from '../lib/axios';
import { useAuth } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';
import Checkbox from '@mui/material/Checkbox';
import ListItemText from '@mui/material/ListItemText';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
  Alert
} from '@mui/material';

interface Dataset {
  id: number;
  filename: string;
  file_type: string;
  created_at: string;
}

const categoricalAlgorithms = [
  { value: 'label', label: 'Label Encoding' },
  { value: 'one_hot', label: 'OneHot Encoding' }
];

const scalingAlgorithms = [
  { value: 'robust', label: 'Robust Scaler (Non Normal)' },
  { value: 'standard', label: 'Z-score Standardization (Normal)' }
];

const DataTransformationPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || undefined;
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [catAlgo, setCatAlgo] = useState<string>('label');
  const [scalingAlgo, setScalingAlgo] = useState<string>('robust');
  const [loading, setLoading] = useState<boolean>(false);
  const [successMsg, setSuccessMsg] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');

  // Fetch columns when dataset changes
  useEffect(() => {
    if (!selectedDataset || !token) {
      setColumns([]);
      setSelectedColumns([]);
      return;
    }
    setLoading(true);
    api.get(`/datasets/${selectedDataset}`)
      .then(res => {
        let cols: string[] = [];
        if (Array.isArray(res.data.columns)) {
          cols = res.data.columns;
        } else if (res.data.data_types && typeof res.data.data_types === 'object') {
          cols = Object.keys(res.data.data_types);
        }
        setColumns(cols);
        setSelectedColumns([]);
      })
      .catch(() => setErrorMsg("Erreur lors du chargement des colonnes du dataset."))
      .finally(() => setLoading(false));
  }, [selectedDataset, token]);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    api.get('/datasets/')
      .then(res => setDatasets(res.data))
      .catch(() => setErrorMsg("Erreur lors du chargement des datasets."))
      .finally(() => setLoading(false));
  }, [token]);

  const handleTransform = async () => {
    if (!selectedDataset) {
      setErrorMsg('Veuillez choisir un dataset.');
      return;
    }
    setLoading(true);
    setSuccessMsg('');
    setErrorMsg('');
    if (!selectedColumns.length) {
      setErrorMsg('Veuillez sélectionner au moins une colonne.');
      setLoading(false);
      return;
    }
    try {
      const config: any = {
        categorical_encoding: { methods: [{ method: catAlgo, columns: selectedColumns }] },
        feature_scaling: { methods: [{ method: scalingAlgo, columns: selectedColumns }] }
      };
      const url = `/transformation/transform${sessionId ? `?session_id=${sessionId}` : ''}`;
      const response = await api.post(url, {
        dataset_id: Number(selectedDataset),
        config
      });
      setSuccessMsg('Transformation appliquée !');
      // TODO: gérer la réponse (ex: afficher lien de téléchargement)
    } catch (err: any) {
      setErrorMsg('Erreur lors de la transformation.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box maxWidth={600} mx="auto" mt={4}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Data Transformation
          </Typography>
          <Divider sx={{ mb: 2 }} />
          {errorMsg && <Alert severity="error">{errorMsg}</Alert>}
          {successMsg && <Alert severity="success">{successMsg}</Alert>}

          <FormControl fullWidth margin="normal">
            <InputLabel id="dataset-select-label">Choisir un dataset</InputLabel>
            <Select
              labelId="dataset-select-label"
              value={selectedDataset}
              label="Choisir un dataset"
              onChange={e => setSelectedDataset(e.target.value)}
              disabled={loading}
            >
              {datasets.map(ds => (
                <MenuItem key={ds.id} value={ds.id}>{ds.filename}</MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Sélection de colonnes */}
          {columns.length > 0 && (
            <FormControl fullWidth margin="normal">
              <InputLabel id="columns-select-label">Colonnes à transformer</InputLabel>
              <Select
                labelId="columns-select-label"
                multiple
                value={selectedColumns}
                onChange={e => setSelectedColumns(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value as string[])}
                renderValue={selected => (selected as string[]).join(', ')}
                disabled={loading}
              >
                {columns.map(col => (
                  <MenuItem key={col} value={col}>
                    <Checkbox checked={selectedColumns.indexOf(col) > -1} />
                    <ListItemText primary={col} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <Box mt={3}>
            <Typography variant="subtitle1">Handling Categorical Data</Typography>
            <RadioGroup
              row
              value={catAlgo}
              onChange={e => setCatAlgo(e.target.value)}
            >
              {categoricalAlgorithms.map(opt => (
                <FormControlLabel key={opt.value} value={opt.value} control={<Radio />} label={opt.label} />
              ))}
            </RadioGroup>
          </Box>

          <Box mt={3}>
            <Typography variant="subtitle1">Transformation (Standardization)</Typography>
            <RadioGroup
              row
              value={scalingAlgo}
              onChange={e => setScalingAlgo(e.target.value)}
            >
              {scalingAlgorithms.map(opt => (
                <FormControlLabel key={opt.value} value={opt.value} control={<Radio />} label={opt.label} />
              ))}
            </RadioGroup>
          </Box>

          <Box mt={4} textAlign="center">
            <Button
              variant="contained"
              color="primary"
              disabled={loading}
              onClick={handleTransform}
              startIcon={loading ? <CircularProgress size={20} /> : null}
            >
              Appliquer la transformation
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DataTransformationPage;
