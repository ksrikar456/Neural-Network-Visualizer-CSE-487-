export interface GalleryItem {
  id: string;
  timestamp: Date;
  contentImageUrl: string;
  styleImageUrl: string;
  resultImageUrl: string;
  bestLoss: number;
  styleLoss: number;
  contentLoss: number;
  processingTime: number;  // in seconds
  parameters: {
    styleWeight: number;
    contentWeight: number;//test
    numSteps: number;
    layerWeights?: Record<string, number>;
  };
} 