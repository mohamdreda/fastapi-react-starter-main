export type WorkflowStatus = 'queued' | 'running' | 'success' | 'failed';

export interface WorkflowStepRun {
  order: number;
  step: string;
  substep?: string | null;
  algorithm: string;
  params?: Record<string, any> | null;
  status: WorkflowStatus;
  elapsed_ms?: number | null;
  metrics?: Record<string, any> | null;
  visuals?: Array<Record<string, any>> | null;
  error?: string | null;
}

export interface WorkflowRun {
  id: string;
  dataset_id: number;
  template_id?: string | null;
  status: WorkflowStatus;
  started_at?: string | null;
  finished_at?: string | null;
  steps: WorkflowStepRun[];
}

export interface TemplateStep {
  step: string;
  algorithm: string;
  substep?: string | null;
  params?: Record<string, any> | null;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  version: number;
  description?: string | null;
  steps: TemplateStep[];
  created_at?: string | null;
}
