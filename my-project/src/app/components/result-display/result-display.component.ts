import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StyleTransferJob } from '../../models/style-transfer.model';
import { StyleTransferService } from '../../services/style-transfer.service';

@Component({
  selector: 'app-result-display',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="result-container p-6 bg-white rounded-lg shadow-md">
      <h3 class="text-xl font-bold mb-6 text-purple-800">Result</h3>
      
      <!-- Initial state -->
      <div *ngIf="!job" class="text-center py-12">
        <div class="text-gray-400">
          <svg xmlns="http://www.w3.org/2000/svg" class="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <p class="mt-4 text-gray-600">Upload your content and style images, then apply style transfer to see the result</p>
      </div>
      
      <!-- Processing state -->
      <div *ngIf="job && (job.status === 'pending' || job.status === 'processing')" class="text-center py-8">
        <div class="mx-auto">
          <svg class="animate-spin h-12 w-12 text-purple-600 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        </div>
        
        <p class="mt-4 text-gray-600 font-medium">Processing your style transfer...</p>
        
        <div *ngIf="job.progress !== undefined && job.progress !== null" class="mt-6 px-4">
          <div class="relative pt-1">
            <div class="flex mb-2 items-center justify-between">
              <div>
                <span class="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-purple-600 bg-purple-200">
                  Progress
                </span>
              </div>
              <div class="text-right">
                <span class="text-xs font-semibold inline-block text-purple-600">
                  {{ job.progress.toFixed(0) }}%
                </span>
              </div>
            </div>
            <div class="overflow-hidden h-2 mb-4 text-xs flex rounded bg-purple-200">
              <div [style.width.%]="job.progress || 0" class="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-purple-600"></div>
            </div>
          </div>
          
          <div *ngIf="job.styleLoss || job.contentLoss" class="grid grid-cols-2 gap-4 text-sm mt-2">
            <div>
              <div class="text-gray-500">Style Loss</div>
              <div class="font-semibold">{{ job.styleLoss ? job.styleLoss.toFixed(4) : '--' }}</div>
            </div>
            <div>
              <div class="text-gray-500">Content Loss</div>
              <div class="font-semibold">{{ job.contentLoss ? job.contentLoss.toFixed(4) : '--' }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Completed state -->
      <div *ngIf="job && job.status === 'completed' && job.resultUrl" class="py-2">
        <div class="result-image-container">
          <img [src]="getFullResultUrl(job.resultUrl)" alt="Style Transfer Result" 
               class="w-full rounded-lg shadow-lg max-h-[500px] object-contain mx-auto">
        </div>
        
        <div class="mt-4 flex justify-center">
          <a [href]="getFullResultUrl(job.resultUrl)" download="styled_image.jpg" 
             class="bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 focus:outline-none inline-flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download Image
          </a>
        </div>
      </div>
      
      <!-- Error state -->
      <div *ngIf="job && job.status === 'failed'" class="text-center py-8">
        <div class="mx-auto text-red-500">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        
        <p class="mt-4 text-gray-800 font-medium">Style transfer failed</p>
        <p *ngIf="job.error" class="mt-2 text-red-600 text-sm">{{ job.error }}</p>
        
        <p class="mt-4 text-gray-600">Please try again with different images or settings</p>
      </div>
    </div>
  `,
  styles: []
})
export class ResultDisplayComponent {
  @Input() job: StyleTransferJob | null = null;
  
  constructor(private styleTransferService: StyleTransferService) {}
  
  getFullResultUrl(relativeUrl: string): string {
    return this.styleTransferService.getResultUrl(relativeUrl);
  }
} 