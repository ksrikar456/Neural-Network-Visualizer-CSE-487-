import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { StyleTransferParams, StyleLayerInfo } from '../../models/style-transfer.model';
import { StyleTransferService } from '../../services/style-transfer.service';
import { WeightPresetService, WeightPreset } from '../../services/weight-preset.service';

@Component({
  selector: 'app-style-controls',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="controls-container p-6 bg-white rounded-lg shadow-md">
      <h3 class="text-xl font-bold mb-6 text-purple-800">Style Transfer Controls</h3>
      
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Style vs. Content Balance
        </label>
        <div class="slider-container">
          <input type="range" 
                 min="0" 
                 max="6" 
                 step="1" 
                 [(ngModel)]="selectedPresetIndex" 
                 (ngModelChange)="onPresetChange()"
                 class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
        </div>
        <div class="mt-1 grid grid-cols-2">
          <span class="text-xs text-gray-500">More Content</span>
          <span class="text-xs text-gray-500 text-right">More Style</span>
        </div>
        <div class="text-sm text-gray-700 mt-3 text-center">
          {{ weightPresets[selectedPresetIndex].label }}
        </div>
        <div class="text-xs text-gray-500 mt-1 text-center">
          {{ weightPresets[selectedPresetIndex].description }}
        </div>
      </div>
      
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-700 mb-1">
          Optimization Steps: {{ params.numSteps }}
        </label>
        <div class="slider-container">
          <input type="range" 
                 min="50" 
                 max="400" 
                 step="10" 
                 [(ngModel)]="params.numSteps" 
                 (ngModelChange)="onParamsChange()"
                 class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
        </div>
        <div class="text-xs text-gray-500 mt-1">Higher values produce better results but take longer to process</div>
      </div>
      
      <div class="mb-6" *ngIf="showAdvanced">
        <h4 class="text-lg font-semibold mb-3 text-purple-700">Advanced Options</h4>
        
        <!-- Display actual weight values -->
        <div class="bg-gray-100 p-3 mb-4 rounded-md">
          <div class="grid grid-cols-2 gap-2 mb-2">
            <div class="text-sm font-medium text-gray-700">Style Weight:</div>
            <div class="text-sm text-gray-800 text-right">{{ params.styleWeight.toExponential(1) }}</div>
            
            <div class="text-sm font-medium text-gray-700">Content Weight:</div>
            <div class="text-sm text-gray-800 text-right">{{ params.contentWeight.toFixed(1) }}</div>
          </div>
          <p class="text-xs text-gray-500 mt-1">These weights determine how the algorithm balances style and content elements</p>
        </div>
        
        <!-- Layer weights -->
        <h5 class="text-md font-semibold mt-6 mb-3 text-purple-700">Layer Weights</h5>
        <div *ngFor="let layer of styleLayers" class="mb-3">
          <div class="flex justify-between mb-1">
            <label class="text-sm font-medium text-gray-700">{{ layer.name }}</label>
            <span class="text-sm text-gray-500">{{ getLayerWeight(layer.name).toFixed(1) }}</span>
          </div>
          <div class="slider-container">
            <input type="range" 
                   min="0" 
                   max="2" 
                   step="0.1" 
                   [ngModel]="getLayerWeight(layer.name)" 
                   (ngModelChange)="setLayerWeight(layer.name, $event)"
                   class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
          </div>
          <div class="text-xs text-gray-500 mt-1">{{ layer.description }}</div>
        </div>
      </div>
      
      <div class="flex flex-col space-y-2">
        <button type="button" 
                (click)="toggleAdvanced()"
                class="text-purple-700 font-medium py-2 px-4 text-sm rounded-md hover:bg-purple-50 focus:outline-none">
          {{ showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options' }}
        </button>
        
        <button type="button" 
                (click)="resetParams()"
                class="bg-gray-200 text-gray-800 py-2 px-4 rounded-md hover:bg-gray-300 focus:outline-none">
          Reset to Defaults
        </button>
        
        <button type="button" 
                (click)="applyStyleTransfer()"
                [disabled]="!canApply"
                [class.opacity-50]="!canApply"
                [class.cursor-not-allowed]="!canApply"
                class="bg-purple-600 text-white py-3 px-4 rounded-md hover:bg-purple-700 focus:outline-none flex justify-center items-center space-x-2">
          <span>Apply Style Transfer</span>
          <svg *ngIf="processing" class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        </button>
      </div>
    </div>
  `,
  styles: []
})
export class StyleControlsComponent implements OnInit {
  @Input() canApply: boolean = false;
  @Input() processing: boolean = false;
  @Output() apply = new EventEmitter<StyleTransferParams>();
  
  params: StyleTransferParams = {
    styleWeight: 1000000,
    contentWeight: 1.0,
    numSteps: 300,
    layerWeights: {}
  };
  
  // Use the weight presets from the service
  weightPresets: WeightPreset[] = [];
  
  selectedPresetIndex: number = 3; // Default to "Perfect Balance"
  showAdvanced: boolean = false;
  styleLayers: StyleLayerInfo[] = [];
  
  constructor(
    private styleTransferService: StyleTransferService,
    private weightPresetService: WeightPresetService
  ) {}
  
  ngOnInit(): void {
    this.styleLayers = this.styleTransferService.getStyleLayers();
    this.weightPresets = this.weightPresetService.getAllPresets();
    this.resetParams();
  }
  
  onPresetChange(): void {
    const preset = this.weightPresets[this.selectedPresetIndex];
    this.params.styleWeight = preset.styleWeight;
    this.params.contentWeight = preset.contentWeight;
    this.onParamsChange();
  }
  
  toggleAdvanced(): void {
    this.showAdvanced = !this.showAdvanced;
  }
  
  onParamsChange(): void {
    // Optional: Could implement validation or other logic here
  }
  
  resetParams(): void {
    // Reset to the default preset (Perfect Balance)
    this.selectedPresetIndex = 3;
    
    const preset = this.weightPresets[this.selectedPresetIndex];
    this.params = {
      styleWeight: preset.styleWeight,
      contentWeight: preset.contentWeight,
      numSteps: 300,
      layerWeights: {}
    };
    
    // Reset layer weights to defaults
    this.styleLayers.forEach(layer => {
      if (!this.params.layerWeights) {
        this.params.layerWeights = {};
      }
      this.params.layerWeights[layer.name] = layer.defaultWeight;
    });
    
    this.onParamsChange();
  }
  
  getLayerWeight(layerName: string): number {
    if (!this.params.layerWeights) {
      return 1.0;
    }
    return this.params.layerWeights[layerName] || 1.0;
  }
  
  setLayerWeight(layerName: string, weight: number): void {
    if (!this.params.layerWeights) {
      this.params.layerWeights = {};
    }
    this.params.layerWeights[layerName] = weight;
    this.onParamsChange();
  }
  
  applyStyleTransfer(): void {
    if (this.canApply && !this.processing) {
      this.apply.emit({...this.params});
    }
  }
} 