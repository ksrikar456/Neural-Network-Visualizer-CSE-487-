export interface StyleTransferParams {
  styleWeight: number;
  contentWeight: number;
  numSteps: number;
  layerWeights?: Record<string, number>;
}

export interface ImageFile {
  file: File | null;
  preview: string | null;
}

export interface StyleTransferJob {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  styleLoss?: number;
  contentLoss?: number;
  resultUrl?: string;
  error?: string;
}

export interface StyleLayerInfo {
  name: string;
  description: string;
  defaultWeight: number;
} 