import React, { useState, useEffect, useRef, useCallback } from 'react';
import { toPng, toBlob } from 'html-to-image';
import { WorkflowRun, WorkflowTemplate, WorkflowStepRun, TemplateStep } from '@/types/workflow';
import { useAuth } from '@/context/AuthContext';
import api from '@/lib/axios';

interface Dataset {
  id: number;
  filename: string;
}

const WorkflowPage: React.FC = () => {
  const { token } = useAuth();
  const workflowContainerRef = useRef<HTMLDivElement>(null);
  // Inner content ref: we export this (full width) instead of the scroll container to avoid cropping
  const workflowContentRef = useRef<HTMLDivElement>(null);
  const sessionId = localStorage.getItem('active_session_id') || '';

  // États pour la sélection et l'affichage
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runsForDataset, setRunsForDataset] = useState<WorkflowRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowTemplate | null>(null);
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [sessionSteps, setSessionSteps] = useState<any[]>([]);
  const [templateName, setTemplateName] = useState<string>('');
  const [templateDesc, setTemplateDesc] = useState<string>('');
  const [savingTemplate, setSavingTemplate] = useState<boolean>(false);
  const [selectedStepIds, setSelectedStepIds] = useState<string[]>([]);
  const hasUserEditedSelection = useRef(false);

  // États pour l'UI
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [templateError, setTemplateError] = useState<string | null>(null);

  // État de l'éditeur de workflow (pour le template sélectionné)
  const [editMode, setEditMode] = useState<boolean>(false);
  const [editableTemplateName, setEditableTemplateName] = useState<string>('');
  const [editableTemplateDesc, setEditableTemplateDesc] = useState<string>('');
  const [editableSteps, setEditableSteps] = useState<TemplateStep[]>([]);

  useEffect(() => {
    const fetchDatasets = async () => {
      if (!token) {
        setErrorMsg('Vous devez être connecté pour accéder aux workflows.');
        return;
      }

      setLoading(true);
      setErrorMsg(null);
      try {
        const response = await api.get<Dataset[]>('datasets/');
        setDatasets(response.data);
      } catch (err) {
        setErrorMsg("Erreur lors du chargement des datasets.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchDatasets();
  }, [token]);

  // Charger les templates disponibles
  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const { data } = await api.get<WorkflowTemplate[]>('workflows/templates');
        setTemplates(data || []);
      } catch (e) {
        // non-bloquant
      }
    })();
  }, [token]);

  // Charger tous les runs (sans filtre dataset)
  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const { data } = await api.get<WorkflowRun[]>('workflows/runs');
        setRunsForDataset(data || []);
      } catch (e) {
        // non-bloquant
      }
    })();
  }, [token]);

  // Sélectionner un run et charger son template pour afficher la configuration
  const handleSelectRun = async (run: WorkflowRun) => {
    setSelectedRun(run);
    setTemplateError(null);
    setSelectedTemplate(null);
    if (!run?.template_id) return;
    try {
      const { data: templates } = await api.get<WorkflowTemplate[]>('workflows/templates');
      const tmpl = (templates || []).find((t) => t.id === run.template_id) || null;
      setSelectedTemplate(tmpl);
      if (!tmpl) setTemplateError('Template introuvable pour ce run.');
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message;
      setTemplateError(`Impossible de charger la configuration du workflow${status ? ` (HTTP ${status})` : ''}. ${detail ?? ''}`);
    }
  };

  // Handler: enregistrer un template à partir de la session active
  const handleSaveTemplateFromSession = async () => {
    if (!sessionId) {
      setErrorMsg('Aucune session active.');
      return;
    }
    if (!templateName.trim()) {
      setErrorMsg('Veuillez saisir un nom pour le workflow.');
      return;
    }
    setSavingTemplate(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const payload = {
        name: templateName.trim(),
        description: templateDesc || undefined,
        selected_step_ids: selectedStepIds && selectedStepIds.length > 0 ? selectedStepIds : undefined,
      };
      const { data } = await api.post(`workflows/templates/from-session/${sessionId}`, payload);
      setSuccessMsg(`Workflow enregistré comme template: ${data.name} (v${data.version}).`);
      // Afficher immédiatement dans la liste locale sans recharger
      setTemplates((prev) => [data, ...prev]);
      setTemplateName('');
      setTemplateDesc('');
      // Ouvrir l'éditeur sur ce template nouvellement créé
      setSelectedTemplate(data);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message;
      setErrorMsg(`Échec de l'enregistrement du workflow${status ? ` (HTTP ${status})` : ''}. ${detail ?? ''}`);
    } finally {
      setSavingTemplate(false);
    }
  };

  // Poll des étapes de session si une session active existe
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      try {
        const { data } = await api.get<any[]>(`sessions/${sessionId}/steps`);
        setSessionSteps(data || []);
      } catch (e) {
        // non-bloquant
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [sessionId]);

  // Synchroniser la sélection sans écraser les choix de l'utilisateur :
  // - Avant toute interaction utilisateur: tout sélectionner une fois, puis ajouter les nouvelles étapes
  // - Après interaction: ne jamais réajouter automatiquement; uniquement supprimer les étapes disparues
  useEffect(() => {
    const currentIds: string[] = (sessionSteps || []).map((s: any) => s.id).filter(Boolean);
    setSelectedStepIds((prev) => {
      const previous = Array.isArray(prev) ? prev : [];
      // Toujours élaguer les étapes qui n'existent plus
      const pruned = previous.filter((id) => currentIds.includes(id));

      if (!hasUserEditedSelection.current) {
        if (pruned.length === 0) {
          // Hydratation initiale: sélectionner tout
          return currentIds;
        }
        // Avant interaction: ajouter uniquement les nouvelles étapes
        const toAdd = currentIds.filter((id) => !pruned.includes(id));
        if (toAdd.length > 0) {
          return [...pruned, ...toAdd];
        }
      }
      return pruned;
    });
  }, [sessionSteps]);

  // Utilitaires de sélection d'étapes pour la prévisualisation
  const toggleStep = (id: string) => {
    hasUserEditedSelection.current = true;
    setSelectedStepIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };
  const selectAllSteps = () => {
    hasUserEditedSelection.current = true;
    const ids = (sessionSteps || []).map((s: any) => s.id).filter(Boolean);
    setSelectedStepIds(ids);
  };
  const deselectAllSteps = () => {
    hasUserEditedSelection.current = true;
    setSelectedStepIds([]);
  };

  // Poll du run sélectionné pour actualiser statut et étapes
  useEffect(() => {
    if (!selectedRun?.id) return;
    const interval = setInterval(async () => {
      try {
        const { data: freshRun } = await api.get<WorkflowRun>(`workflows/runs/${selectedRun.id}`);
        let steps: WorkflowStepRun[] = [];
        try {
          const { data: s } = await api.get<WorkflowStepRun[]>(`workflows/runs/${selectedRun.id}/steps`);
          steps = s || [];
        } catch {
          // si l'endpoint steps n'est pas dispo, ignorer
        }
        const merged = { ...freshRun, steps } as WorkflowRun;
        setSelectedRun(merged);
        // mettre à jour la liste
        setRunsForDataset((prev) => prev.map((r) => (r.id === merged.id ? merged : r)));
        if (freshRun.status === 'success' || freshRun.status === 'failed') {
          clearInterval(interval);
        }
      } catch (e) {
        // ignorer erreurs transitoires
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedRun?.id]);

  // Fonction pour obtenir les classes CSS du badge selon le statut
  const getStatusBadgeClass = (status: string) => {
    const baseClasses = 'px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide';

    switch (status) {
      case 'success':
        return `${baseClasses} bg-green-500 text-white`;
      case 'running':
        return `${baseClasses} bg-blue-500 text-white`;
      case 'failed':
        return `${baseClasses} bg-red-500 text-white`;
      case 'queued':
      default:
        return `${baseClasses} bg-gray-500 text-white`;
    }
  };

  // Fonction pour formater la durée
  const formatElapsedTime = (elapsedMs?: number) => {
    if (!elapsedMs) return '';
    const seconds = Math.floor(elapsedMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return minutes > 0 ? `${minutes}m ${remainingSeconds}s` : `${remainingSeconds}s`;
  };

  // Friendly label for session steps: "Tool: Step" (e.g., "Data Diagnosis: Analysis")
  const formatStepLabel = (s: any) => {
    const toolRaw = (s?.tool || '').toString();
    const stepRaw = (s?.step || s?.name || '').toString();
    const toolTitleMap: Record<string, string> = {
      diagnosis: 'Data Diagnosis',
      data_diagnostic: 'Data Diagnosis',
      cleaning: 'Data Cleaning',
      resolution: 'Resolution',
      profiling: 'Profiling',
    };
    const toTitleCase = (str: string) => str.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    const toolTitle = toolTitleMap[toolRaw] || (toolRaw ? toTitleCase(toolRaw) : '');
    if (toolTitle && stepRaw) return `${toolTitle}: ${toTitleCase(stepRaw)}`;
    return stepRaw || toolTitle || 'Step';
  };

  // Formatter for template steps in the editor (TemplateStep has optional 'tool')
  const formatTemplateStepLabel = (st: TemplateStep) => {
    return formatStepLabel(st as any);
  };

  // Friendly display names for encoding/scaling methods
  const friendlyEncodingName = (raw?: string) => {
    const k = (raw || '').toLowerCase().replace(/\s+/g, '_');
    const map: Record<string, string> = {
      label: 'Label Encoding',
      label_encoding: 'Label Encoding',
      one_hot: 'One-Hot Encoding',
      onehot: 'One-Hot Encoding',
      one_hot_encoding: 'One-Hot Encoding',
      ordinal: 'Ordinal Encoding',
      target: 'Target Encoding',
      frequency: 'Frequency Encoding',
      binary: 'Binary Encoding',
      hash: 'Hashing Encoding',
      hashing: 'Hashing Encoding',
    };
    return map[k] || (raw ? raw : 'Method');
  };

  const friendlyScalingName = (raw?: string) => {
    const k = (raw || '').toLowerCase().replace(/\s+/g, '_');
    const map: Record<string, string> = {
      standardization: 'Standardization',
      standard_scaler: 'Standardization',
      minmax: 'Min-Max Scaling',
      min_max: 'Min-Max Scaling',
      min_max_scaler: 'Min-Max Scaling',
      normalization: 'Normalization',
      normalize: 'Normalization',
      robust: 'Robust Scaling',
      robust_scaler: 'Robust Scaling',
    };
    return map[k] || (raw ? raw : 'Scaling');
  };

  // Renders a concise summary for categorical encoding configuration if present
  const renderCategoricalSummary = (cfg: any) => {
    const ce = cfg?.categorical_encoding;
    const methods: Array<{ method?: string; columns?: string[] }> = Array.isArray(ce?.methods)
      ? ce.methods
      : [];
    if (!methods.length) return null;
    return (
      <div className="mt-1 space-y-1">
        <div className="text-[11px] text-gray-500">Categorical Encoding</div>
        <ul className="pl-3 list-disc space-y-1">
          {methods.map((m, i) => (
            <li key={i} className="text-xs text-gray-700">
              <span className="inline-flex items-center px-2 py-0.5 mr-2 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">
                {friendlyEncodingName(m?.method)}
                {Array.isArray(m?.columns) && m.columns!.length > 0 ? ` (${m.columns!.length} cols)` : ''}
              </span>
              {Array.isArray(m?.columns) && m!.columns!.length > 0 && (
                <span className="inline-flex flex-wrap gap-1 align-middle">
                  {m!.columns!.map((c, idx) => (
                    <span key={idx} className="inline-block px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">
                      {c}
                    </span>
                  ))}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Renders a concise summary for feature scaling configuration if present
  const renderFeatureScalingSummary = (cfg: any) => {
    const fs = cfg?.feature_scaling;
    const methods: Array<{ method?: string; columns?: string[] }> = Array.isArray(fs?.methods)
      ? fs.methods
      : [];
    if (!methods.length) return null;
    return (
      <div className="mt-2 space-y-1">
        <div className="text-[11px] text-gray-500">Feature Scaling</div>
        <ul className="pl-3 list-disc space-y-1">
          {methods.map((m, i) => (
            <li key={i} className="text-xs text-gray-700">
              <span className="inline-flex items-center px-2 py-0.5 mr-2 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">
                {friendlyScalingName(m?.method)}
                {Array.isArray(m?.columns) && m.columns!.length > 0 ? ` (${m.columns!.length} cols)` : ''}
              </span>
              {Array.isArray(m?.columns) && m!.columns!.length > 0 && (
                <span className="inline-flex flex-wrap gap-1 align-middle">
                  {m!.columns!.map((c, idx) => (
                    <span key={idx} className="inline-block px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">
                      {c}
                    </span>
                  ))}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Combines summaries and hides when nothing is used
  const renderConfigSummary = (cfg: any) => {
    const cat = renderCategoricalSummary(cfg);
    const scale = renderFeatureScalingSummary(cfg);
    if (!cat && !scale) return null;
    return (
      <div className="mt-1">
        {cat}
        {scale}
      </div>
    );
  };

  // Deduplication helpers (Similarity / Classification / Clustering / Result)
  const friendlyModelName = (raw?: string) => {
    const k = (raw || '').toLowerCase();
    const map: Record<string, string> = {
      random_forest: 'Random Forest',
      logistic_regression: 'Logistic Regression',
      xgboost: 'XGBoost',
      svm: 'SVM',
      knn: 'k-NN',
    };
    return map[k] || (raw ? raw : 'Model');
  };

  const friendlyClusterMethod = (raw?: string) => {
    const k = (raw || '').toLowerCase();
    const map: Record<string, string> = {
      graph_connected_components: 'Graph Connected Components',
      dbscan: 'DBSCAN',
      hdbscan: 'HDBSCAN',
    };
    return map[k] || (raw ? raw : 'Method');
  };

  const friendlyResultMethod = (raw?: string) => {
    const k = (raw || '').toLowerCase();
    const map: Record<string, string> = {
      keep_first: 'Keep First',
      keep_longest: 'Keep Longest',
      keep_best: 'Keep Best',
    };
    return map[k] || (raw ? raw : 'Method');
  };

  const renderSimilarityFieldConfigs = (fieldConfigs: any) => {
    if (!fieldConfigs || typeof fieldConfigs !== 'object') return null;
    const entries = Object.entries(fieldConfigs);
    if (entries.length === 0) return null;
    const maxPreview = 3;
    const preview = entries.slice(0, maxPreview);
    const remaining = entries.length - preview.length;
    return (
      <div className="mt-1 space-y-1">
        <div className="text-[11px] text-gray-500">Field configs</div>
        <ul className="pl-3 list-disc space-y-1">
          {preview.map(([field, cfg]: any, i: number) => (
            <li key={i} className="text-xs text-gray-700">
              <span className="inline-flex items-center px-2 py-0.5 mr-2 rounded bg-blue-50 text-blue-700 border border-blue-100">{field}</span>
              {cfg?.method && (
                <span className="inline-flex items-center px-2 py-0.5 mr-2 rounded bg-gray-100 text-gray-800 border border-gray-200">{String(cfg.method)}</span>
              )}
              {cfg?.weight !== undefined && (
                <span className="inline-flex items-center px-2 py-0.5 mr-2 rounded bg-amber-50 text-amber-700 border border-amber-100">w={String(cfg.weight)}</span>
              )}
            </li>
          ))}
        </ul>
        {remaining > 0 && (
          <div className="text-[11px] text-gray-500">+{remaining} more field(s)</div>
        )}
      </div>
    );
  };

  const renderDedupSummary = (tool: string, params: any) => {
    const key = (tool || '').toLowerCase();
    if (!params || typeof params !== 'object') return null;
    if (key === 'classification') {
      const m = friendlyModelName(params?.method || params?.model);
      const chips: string[] = [];
      if (params?.n_estimators !== undefined) chips.push(`n_estimators: ${params.n_estimators}`);
      if (params?.max_depth !== undefined) chips.push(`max_depth: ${params.max_depth}`);
      if (params?.confidence_threshold !== undefined) chips.push(`confidence: ${params.confidence_threshold}`);
      return (
        <div className="space-y-1">
          <div className="text-[11px] text-gray-500">Classification</div>
          <div className="flex flex-wrap gap-1">
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-100">{m}</span>
            {chips.map((c, i) => (
              <span key={i} className="inline-flex items-center px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">{c}</span>
            ))}
          </div>
        </div>
      );
    }
    if (key === 'clustering') {
      const m = friendlyClusterMethod(params?.method);
      const chips: string[] = [];
      if (params?.confidence_threshold !== undefined) chips.push(`confidence: ${params.confidence_threshold}`);
      return (
        <div className="space-y-1">
          <div className="text-[11px] text-gray-500">Clustering</div>
          <div className="flex flex-wrap gap-1">
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-100">{m}</span>
            {chips.map((c, i) => (
              <span key={i} className="inline-flex items-center px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">{c}</span>
            ))}
          </div>
        </div>
      );
    }
    if (key === 'result') {
      const m = friendlyResultMethod(params?.method);
      return (
        <div className="space-y-1">
          <div className="text-[11px] text-gray-500">Result</div>
          <div className="flex flex-wrap gap-1">
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-slate-50 text-slate-700 border border-slate-100">{m}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  // Extract dataset id from common shapes
  const extractDatasetId = (params: any): string | number | null => {
    if (!params) return null;
    return (
      params.dataset_id ??
      params.datasetId ??
      (typeof params.dataset === 'object' && params.dataset ? params.dataset.id : null)
    );
  };

  // Hydrater l'éditeur dès qu'un template est sélectionné
  useEffect(() => {
    if (selectedTemplate) {
      setEditableTemplateName(selectedTemplate.name || '');
      setEditableTemplateDesc(selectedTemplate.description || '');
      setEditableSteps(Array.isArray(selectedTemplate.steps) ? [...selectedTemplate.steps] : []);
      setEditMode(false);
    }
  }, [selectedTemplate]);

  // Exporter le diagramme en PNG (également disponible par clic sur le diagramme en mode lecture)
  const handleExportPng = useCallback(async () => {
    // Prefer the inner content (full scrollable width/height)
    const node = (workflowContentRef.current || workflowContainerRef.current) as HTMLElement | null;
    if (!node) {
      alert('Aucun contenu à exporter.');
      return;
    }
    // Compute full content dimensions to include all horizontally scrolled steps
    const width = Math.min(node.scrollWidth || node.offsetWidth || 0, 12000); // hard cap to avoid OOM
    const height = Math.min(node.scrollHeight || node.offsetHeight || 0, 8000);
    const options = { cacheBust: true, pixelRatio: 2, width, height, backgroundColor: '#ffffff' } as const;
    try {
      const dataUrl = await toPng(node, options);
      const link = document.createElement('a');
      link.download = `${editableTemplateName || 'workflow'}.png`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('[Workflow] toPng failed, trying toBlob fallback:', err);
      try {
        const blob = await toBlob(node, options);
        if (!blob) throw new Error('Blob is null');
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${editableTemplateName || 'workflow'}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } catch (err2) {
        console.error('[Workflow] toBlob fallback failed:', err2);
        alert("Échec de l'export PNG. Consultez la console pour les détails.");
      }
    }
  }, [workflowContainerRef, editableTemplateName]);

  // Exporter le template/workflow en JSON
  const handleExportTemplateJson = useCallback(() => {
    try {
      const payload = selectedTemplate && selectedTemplate.steps?.length
        ? selectedTemplate
        : {
            name: (editableTemplateName || 'workflow').trim(),
            description: editableTemplateDesc || undefined,
            steps: editableSteps,
          };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${(editableTemplateName || selectedTemplate?.name || 'workflow').trim()}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('[Workflow] JSON export failed:', e);
      alert("Échec de l'export JSON.");
    }
  }, [selectedTemplate, editableTemplateName, editableTemplateDesc, editableSteps]);

  // Helpers d'édition
  const moveStepUp = (idx: number) => {
    if (idx <= 0) return;
    setEditableSteps((prev) => {
      const arr = [...prev];
      [arr[idx - 1], arr[idx]] = [arr[idx], arr[idx - 1]];
      return arr;
    });
  };
  const moveStepDown = (idx: number) => {
    setEditableSteps((prev) => {
      if (idx >= prev.length - 1) return prev;
      const arr = [...prev];
      [arr[idx + 1], arr[idx]] = [arr[idx], arr[idx + 1]];
      return arr;
    });
  };
  const updateStepField = (idx: number, field: keyof TemplateStep, value: any) => {
    setEditableSteps((prev) => prev.map((s, i) => (i === idx ? { ...s, [field]: value } : s)));
  };
  const removeStep = (idx: number) => {
    setEditableSteps((prev) => prev.filter((_, i) => i !== idx));
  };
  const addStep = () => {
    setEditableSteps((prev) => [
      ...prev,
      { tool: '', step: 'step_name', algorithm: 'algorithm_name', substep: undefined, params: {} },
    ]);
  };

  const handleSaveEditedTemplate = async () => {
    try {
      setErrorMsg(null);
      setSuccessMsg(null);
      const payload = {
        name: (editableTemplateName || 'workflow').trim(),
        description: editableTemplateDesc || undefined,
        steps: editableSteps,
      };
      const { data } = await api.post<WorkflowTemplate>('workflows/templates', payload);
      setSuccessMsg(`Template sauvegardé: ${data.name} (v${data.version}).`);
      setTemplates((prev) => [data, ...prev]);
      setSelectedTemplate(data);
      setEditMode(false);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message;
      setErrorMsg(`Échec de la sauvegarde du template${status ? ` (HTTP ${status})` : ''}. ${detail ?? ''}`);
    }
  };

  const handleUpdateExistingTemplate = async () => {
    if (!selectedTemplate?.id) {
      setErrorMsg('Aucun template sélectionné.');
      return;
    }
    try {
      setErrorMsg(null);
      setSuccessMsg(null);
      const payload = {
        name: (editableTemplateName || 'workflow').trim(),
        description: editableTemplateDesc || undefined,
        steps: editableSteps,
      };
      const { data } = await api.put<WorkflowTemplate>(`workflows/templates/${selectedTemplate.id}`, payload);
      setSuccessMsg(`Template mis à jour: ${data.name} (v${data.version}).`);
      setTemplates((prev) => prev.map((t) => (t.id === data.id ? data : t)));
      setSelectedTemplate(data);
      setEditMode(false);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message;
      setErrorMsg(`Échec de la mise à jour du template${status ? ` (HTTP ${status})` : ''}. ${detail ?? ''}`);
    }
  };

  const handleDeleteTemplate = async (id: string) => {
    if (!id) return;
    const ok = window.confirm('Supprimer ce template ? Cette action est irréversible.');
    if (!ok) return;
    try {
      setErrorMsg(null);
      setSuccessMsg(null);
      await api.delete(`workflows/templates/${id}`);
      setTemplates((prev) => prev.filter((t) => t.id !== id));
      if (selectedTemplate?.id === id) {
        setSelectedTemplate(null);
        setEditMode(false);
        setEditableSteps([]);
        setEditableTemplateName('');
        setEditableTemplateDesc('');
      }
      setSuccessMsg('Template supprimé.');
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message;
      setErrorMsg(`Échec de la suppression du template${status ? ` (HTTP ${status})` : ''}. ${detail ?? ''}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Historique des Workflows</h1>

        {/* Messages d'erreur et de succès */}
        {errorMsg && (
          <div className="mb-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
            {errorMsg}
          </div>
        )}
        {successMsg && (
          <div className="mb-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
            {successMsg}
          </div>
        )}

        {/* Générer un Workflow depuis la Session active */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Générer un Workflow depuis la Session</h2>

          {sessionId ? (
            <>
              <div className="text-xs text-gray-500 mb-4">
                Session active: <span className="font-mono">{sessionId}</span> · Étapes: {sessionSteps.length}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Nom du Workflow</label>
                  <input
                    type="text"
                    value={templateName}
                    onChange={(e) => setTemplateName(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="Ex: Nettoyage Session du 20/08"
                  />
                </div>
                <div className="md:col-span-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Description (optionnel)</label>
                  <input
                    type="text"
                    value={templateDesc}
                    onChange={(e) => setTemplateDesc(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="Brève description"
                  />
                </div>
                <div className="md:col-span-1 flex items-end">
                  <button
                    onClick={handleSaveTemplateFromSession}
                    disabled={savingTemplate || !templateName.trim()}
                    className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors w-full"
                  >
                    {savingTemplate ? "Enregistrement..." : "Générer et Enregistrer"}
                  </button>
                </div>
              </div>

              {/* Prévisualisation et sélection des étapes */}
              <div className="mt-6 border-t pt-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Prévisualisation du Workflow (sélectionnez les étapes)</h3>
                <div className="flex gap-2 mb-3">
                  <button onClick={selectAllSteps} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">Tout sélectionner</button>
                  <button onClick={deselectAllSteps} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">Tout désélectionner</button>
                </div>
                {sessionSteps.length === 0 ? (
                  <div className="text-sm text-gray-500">Aucune étape capturée dans la session.</div>
                ) : (
                  <ul className="space-y-2">
                    {sessionSteps.map((s: any, idx: number) => {
                      const included = selectedStepIds.includes(s.id);
                      return (
                        <li key={s.id || idx} className={`p-3 border border-gray-200 rounded ${included ? '' : 'opacity-50'}`}>
                          <div className="flex items-center gap-3">
                            <input
                              type="checkbox"
                              checked={included}
                              onChange={() => toggleStep(s.id)}
                            />
                            <span className="font-medium">{formatStepLabel(s)}</span>
                            {s.algorithm && (
                              <span className="text-xs text-purple-700 bg-purple-50 px-2 py-0.5 rounded">{s.algorithm}</span>
                            )}
                            <span
                              className={`ml-auto px-2 py-0.5 rounded text-white text-xs ${
                                s.status === 'success'
                                  ? 'bg-green-600'
                                  : s.status === 'failed'
                                  ? 'bg-red-600'
                                  : s.status === 'running'
                                  ? 'bg-blue-600'
                                  : 'bg-gray-600'
                              }`}
                            >
                              {s.status || '—'}
                            </span>
                          </div>
                          {extractDatasetId(s.params) !== null && (
                            <div className="mt-1 text-[11px] text-gray-500">
                              Dataset:{' '}
                              <span className="inline-block px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">
                                {String(extractDatasetId(s.params))}
                              </span>
                            </div>
                          )}
                          {s.params && Object.keys(s.params || {}).length > 0 && (
                            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-700">
                              {Object.entries(s.params)
                                .filter(([_, v]) => v !== null && v !== undefined)
                                .map(([k, v]) => {
                                  const tool = (s?.tool || s?.step || '').toLowerCase();
                                  // Deduplication: replace noisy JSON with summaries
                                  if (k === 'field_configs' && tool === 'similarity' && typeof v === 'object') {
                                    return (
                                      <div key={k} className="break-all">{renderSimilarityFieldConfigs(v)}</div>
                                    );
                                  }
                                  if (k === 'params' && ['classification','clustering','result','similarity'].includes(tool) && typeof v === 'object') {
                                    return (
                                      <div key={k} className="break-all">{renderDedupSummary(tool, v)}</div>
                                    );
                                  }
                                  if (k === 'method' && (tool === 'classification' || tool === 'clustering' || tool === 'result')) {
                                    const friendly = tool === 'classification' ? friendlyModelName(String(v)) : tool === 'clustering' ? friendlyClusterMethod(String(v)) : friendlyResultMethod(String(v));
                                    return (
                                      <div key={k} className="break-all">
                                        <span className="font-semibold">{k}:</span> {friendly}
                                      </div>
                                    );
                                  }
                                  if (k === 'config' && typeof v === 'object') {
                                    return (
                                      <div key={k} className="break-all">
                                        {renderConfigSummary(v)}
                                      </div>
                                    );
                                  }
                                  return (
                                    <div key={k} className="break-all">
                                      <span className="font-semibold">{k}:</span>{' '}
                                      {typeof v === 'object' && v !== null ? (
                                        <pre className="mt-1 p-2 bg-gray-50 border rounded overflow-auto max-h-40 whitespace-pre-wrap">
                                          {JSON.stringify(v, null, 2)}
                                        </pre>
                                      ) : (
                                        String(v)
                                      )}
                                    </div>
                                  );
                                })}
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <div className="text-sm text-gray-600">Aucune session active. Connectez-vous pour ouvrir une session.</div>
          )}
        </div>

        {/* Templates enregistrés */}
        {templates.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Templates enregistrés ({templates.length})</h2>
            <div className="space-y-3">
              {templates.map((t) => (
                <div
                  key={t.id}
                  className={`border rounded-lg p-4 flex items-center justify-between cursor-pointer hover:bg-gray-50 ${
                    selectedTemplate?.id === t.id ? 'ring-2 ring-indigo-400' : ''
                  }`}
                  onClick={() => setSelectedTemplate(t)}
                >
                  <div>
                    <div className="font-medium text-gray-900">
                      {t.name}
                      <span className="ml-2 inline-block px-2 py-0.5 text-xs rounded bg-purple-100 text-purple-800">v{t.version}</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {t.created_at ? new Date(t.created_at).toLocaleString() : '—'} · {t.steps?.length ?? 0} étape(s)
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDeleteTemplate(t.id); }}
                      className="px-2 py-1 text-xs rounded bg-red-50 text-red-700 hover:bg-red-100"
                    >
                      Supprimer
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Visualisation / Édition du workflow sélectionné */}
        {selectedTemplate && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Éditeur de Workflow</h2>
              <div className="flex gap-2">
                <button onClick={handleExportPng} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">Télécharger PNG</button>
                <button onClick={handleExportTemplateJson} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">Télécharger JSON</button>
                <button onClick={() => setEditMode((v) => !v)} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">
                  {editMode ? "Terminer l'édition" : 'Modifier'}
                </button>
                {editMode && (
                  <>
                    <button onClick={addStep} className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200">Ajouter une étape</button>
                    <button onClick={handleUpdateExistingTemplate} className="px-3 py-1.5 text-sm rounded bg-green-600 text-white hover:bg-green-700">Mettre à jour le template</button>
                    <button onClick={handleSaveEditedTemplate} className="px-3 py-1.5 text-sm rounded bg-indigo-600 text-white hover:bg-indigo-700">Enregistrer comme nouveau template</button>
                  </>
                )}
              </div>
            </div>

            {/* En-tête (nom / description) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="md:col-span-1">
                <label className="block text-sm font-medium text-gray-700 mb-2">Nom</label>
                <input
                  disabled={!editMode}
                  type="text"
                  value={editableTemplateName}
                  onChange={(e) => setEditableTemplateName(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-100"
                  placeholder="Nom du template"
                />
              </div>
              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
                <input
                  disabled={!editMode}
                  type="text"
                  value={editableTemplateDesc}
                  onChange={(e) => setEditableTemplateDesc(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:bg-gray-100"
                  placeholder="Description du template"
                />
              </div>
            </div>

            {/* Diagramme simple (cliquez pour télécharger si non en mode édition) */}
            <div
              ref={workflowContainerRef}
              className="overflow-x-auto"
              onClick={!editMode ? handleExportPng : undefined}
              title={!editMode ? 'Cliquez pour télécharger en PNG' : undefined}
            >
              <div ref={workflowContentRef} className="min-w-full flex items-stretch gap-4">
                {editableSteps.length === 0 ? (
                  <div className="text-sm text-gray-500">Aucune étape.</div>
                ) : (
                  editableSteps.map((st, idx) => (
                    <div key={idx} className="flex items-stretch">
                      <div className="w-64 border rounded-lg p-3 bg-white shadow-sm">
                        <div className="text-xs text-gray-500 mb-1">Étape {idx + 1}</div>
                        {editMode ? (
                          <div className="space-y-2">
                            <div>
                              <label className="block text-xs text-gray-600">Tool</label>
                              <input value={(st as any).tool || ''} onChange={(e) => updateStepField(idx, 'tool' as any, e.target.value)} className="w-full p-2 border rounded" />
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">Step</label>
                              <input value={st.step} onChange={(e) => updateStepField(idx, 'step', e.target.value)} className="w-full p-2 border rounded" />
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">Substep</label>
                              <input value={st.substep || ''} onChange={(e) => updateStepField(idx, 'substep', e.target.value)} className="w-full p-2 border rounded" />
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">Algorithm</label>
                              <input value={st.algorithm} onChange={(e) => updateStepField(idx, 'algorithm', e.target.value)} className="w-full p-2 border rounded" />
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">Params</label>
                              <pre className="text-xs p-2 bg-gray-50 border rounded overflow-auto max-h-24">{JSON.stringify(st.params || {}, null, 2)}</pre>
                            </div>
                            <div className="flex gap-2">
                              <button onClick={() => moveStepUp(idx)} className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200">Monter</button>
                              <button onClick={() => moveStepDown(idx)} className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200">Descendre</button>
                              <button onClick={() => removeStep(idx)} className="ml-auto px-2 py-1 text-xs rounded bg-red-100 hover:bg-red-200 text-red-700">Supprimer</button>
                            </div>
                          </div>
                        ) : (
                          <div>
                            <div className="font-medium">{formatTemplateStepLabel(st)}</div>
                            {st.algorithm && <div className="text-xs text-gray-600">{st.algorithm}</div>}
                            {st.substep && <div className="text-xs text-gray-600">{st.substep}</div>}
                            {extractDatasetId(st.params) !== null && (
                              <div className="mt-1 text-[11px] text-gray-500">
                                Dataset:{' '}
                                <span className="inline-block px-2 py-0.5 rounded bg-gray-100 text-gray-800 border border-gray-200">
                                  {String(extractDatasetId(st.params))}
                                </span>
                              </div>
                            )}
                            {st.params && Object.keys(st.params).length > 0 && (
                              <div className="mt-2 text-xs text-gray-700">
                                {Object.entries(st.params)
                                  .filter(([_, v]) => v !== null && v !== undefined)
                                  .map(([k, v]) => {
                                    const tool = ((st as any)?.tool || st.step || '').toLowerCase();
                                    if (k === 'field_configs' && tool === 'similarity' && typeof v === 'object') {
                                      return (
                                        <div key={k} className="break-all">{renderSimilarityFieldConfigs(v)}</div>
                                      );
                                    }
                                    if (k === 'params' && ['classification','clustering','result','similarity'].includes(tool) && typeof v === 'object') {
                                      return (
                                        <div key={k} className="break-all">{renderDedupSummary(tool, v)}</div>
                                      );
                                    }
                                    if (k === 'method' && (tool === 'classification' || tool === 'clustering' || tool === 'result')) {
                                      const friendly = tool === 'classification' ? friendlyModelName(String(v)) : tool === 'clustering' ? friendlyClusterMethod(String(v)) : friendlyResultMethod(String(v));
                                      return (
                                        <div key={k} className="break-all">
                                          <span className="font-semibold">{k}:</span> {friendly}
                                        </div>
                                      );
                                    }
                                    if (k === 'config' && typeof v === 'object') {
                                      return (
                                        <div key={k} className="break-all">
                                          {renderConfigSummary(v)}
                                        </div>
                                      );
                                    }
                                    return (
                                      <div key={k} className="break-all">
                                        <span className="font-semibold">{k}:</span>{' '}
                                        {typeof v === 'object' && v !== null ? (
                                          <pre className="mt-1 p-2 bg-gray-50 border rounded overflow-auto max-h-40 whitespace-pre-wrap">
                                            {JSON.stringify(v, null, 2)}
                                          </pre>
                                        ) : (
                                          String(v)
                                        )}
                                      </div>
                                    );
                                  })}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      {idx < editableSteps.length - 1 && (
                        <div className="flex items-center px-2 text-gray-400">→</div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkflowPage;