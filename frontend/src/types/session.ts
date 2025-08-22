export type SessionStepStatus = 'queued' | 'running' | 'success' | 'failed' | string;

export interface SessionCreate {
  title?: string | null;
  description?: string | null;
}

export interface SessionStepRead {
  id: string; // UUID
  order: number;
  tool: string;
  step: string;
  substep?: string | null;
  algorithm?: string | null;
  params?: Record<string, any> | null;
  status: SessionStepStatus;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  run_ref_type?: string | null;
  run_ref_id?: string | null;
}

export interface SessionRead {
  id: string; // UUID
  title?: string | null;
  description?: string | null;
  created_at?: string | null;
  closed_at?: string | null;
  steps: SessionStepRead[];
}

export interface SaveWorkflowFromSessionRequest {
  name: string;
  description?: string | null;
  selected_step_ids?: string[]; // if omitted, include all steps
}
