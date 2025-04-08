import { Injectable } from '@angular/core';

export interface WeightPreset {
  label: string;
  styleWeight: number;
  contentWeight: number;
  description: string;
}

@Injectable({
  providedIn: 'root'
})
export class WeightPresetService {
  // Define the 7 weight presets
  private weightPresets: WeightPreset[] = [
    { 
      label: "Content Focused", 
      styleWeight: 1000, 
      contentWeight: 1,
      description: "Preserves most original content details with minimal style"
    },
    { 
      label: "Content Enhanced", 
      styleWeight: 10000, 
      contentWeight: 3,
      description: "Strong content preservation with light style elements"
    },
    { 
      label: "Balanced (Content)", 
      styleWeight: 100000, 
      contentWeight: 1,
      description: "Balanced transfer with emphasis on content"
    },
    { 
      label: "Perfect Balance", 
      styleWeight: 1000000, 
      contentWeight: 1,
      description: "Equal emphasis on both style and content"
    },
    { 
      label: "Balanced (Style)", 
      styleWeight: 5000000, 
      contentWeight: 1,
      description: "Balanced transfer with emphasis on style"
    },
    { 
      label: "Style Enhanced", 
      styleWeight: 10000000, 
      contentWeight: 1,
      description: "Strong style application while maintaining content structure"
    },
    { 
      label: "Style Focused", 
      styleWeight: 1000000, 
      contentWeight: 0.1,
      description: "Maximizes style effect with minimal content preservation"
    }
  ];

  constructor() { }

  getAllPresets(): WeightPreset[] {
    return this.weightPresets;
  }

  getPresetByIndex(index: number): WeightPreset {
    if (index >= 0 && index < this.weightPresets.length) {
      return this.weightPresets[index];
    }
    return this.weightPresets[3]; // Default to Perfect Balance
  }

  getPresetLabel(styleWeight: number, contentWeight: number): string {
    // Find the closest preset based on the weights
    const preset = this.findClosestPreset(styleWeight, contentWeight);
    return preset ? preset.label : "Custom Balance";
  }

  private findClosestPreset(styleWeight: number, contentWeight: number): WeightPreset | null {
    // Try to find an exact match first
    const exactMatch = this.weightPresets.find(
      preset => preset.styleWeight === styleWeight && preset.contentWeight === contentWeight
    );
    
    if (exactMatch) {
      return exactMatch;
    }

    // Special case for "Style Focused" which has 1e6 style weight but 0.1 content weight
    if (styleWeight === 1000000 && contentWeight <= 0.1) {
      return this.weightPresets[6]; // Style Focused
    }

    // If no exact match, find the closest match based on style weight
    // This is a simplified approach - could be improved with a more sophisticated algorithm
    let closestPreset: WeightPreset | null = null;
    let minDiff = Number.MAX_VALUE;

    for (const preset of this.weightPresets) {
      // Use logarithmic difference for style weight since they span many orders of magnitude
      const styleRatio = Math.abs(Math.log10(styleWeight / preset.styleWeight));
      const contentDiff = Math.abs(contentWeight - preset.contentWeight);
      
      // Weighted sum with more emphasis on style weight
      const diff = styleRatio * 0.7 + contentDiff * 0.3;
      
      if (diff < minDiff) {
        minDiff = diff;
        closestPreset = preset;
      }
    }

    // If the match is very close, return it
    return (minDiff < 1.0) ? closestPreset : null;
  }
} 