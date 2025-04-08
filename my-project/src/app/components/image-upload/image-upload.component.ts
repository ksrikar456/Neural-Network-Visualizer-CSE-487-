import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ImageFile } from '../../models/style-transfer.model';

@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="upload-container">
      <h3 class="text-lg font-semibold mb-2">{{ label }}</h3>
      <div class="upload-box rounded-lg border-2 border-dashed p-4 flex flex-col items-center justify-center"
           [class.border-purple-500]="image?.preview"
           [class.border-gray-300]="!image?.preview"
           [class.bg-purple-50]="image?.preview"
           (click)="fileInput.click()"
           (dragover)="onDragOver($event)"
           (dragleave)="onDragLeave($event)"
           (drop)="onDrop($event)">
        
        <input type="file" 
               #fileInput
               accept="image/*"
               (change)="onFileSelected($event)"
               class="hidden">
        
        <div *ngIf="!image?.preview" class="flex flex-col items-center">
          <div class="upload-icon text-purple-600 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div class="text-gray-600 text-center">
            <div class="font-medium mb-1">Drag & drop or click to upload</div>
            <div class="text-sm text-gray-500">JPG, PNG, GIF files accepted</div>
          </div>
        </div>
        
        <div *ngIf="image?.preview" class="w-full max-h-[250px] overflow-hidden">
          <img [src]="image?.preview || ''" class="w-full h-full object-contain rounded" [alt]="label">
        </div>
      </div>
    </div>
  `,
  styles: [`
    .upload-container {
      margin-bottom: 1.5rem;
    }
    
    .upload-box {
      min-height: 200px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .upload-box:hover {
      background-color: rgba(124, 58, 237, 0.05);
      border-color: #7c3aed;
    }
  `]
})
export class ImageUploadComponent {
  @Input() label: string = 'Upload Image';
  @Input() image: ImageFile | null = null;
  @Output() imageSelected = new EventEmitter<ImageFile>();

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length) {
      this.processFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer?.files && event.dataTransfer.files.length) {
      this.processFile(event.dataTransfer.files[0]);
    }
  }

  private processFile(file: File): void {
    if (file.type.match('image.*')) {
      const reader = new FileReader();
      
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target?.result) {
          const imageFile: ImageFile = {
            file: file,
            preview: e.target.result as string
          };
          
          this.image = imageFile;
          this.imageSelected.emit(imageFile);
        }
      };
      
      reader.readAsDataURL(file);
    }
  }
} 