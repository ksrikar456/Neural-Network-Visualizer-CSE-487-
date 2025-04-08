import { Routes } from '@angular/router';
import { StyleTransferPageComponent } from './pages/style-transfer-page.component';
import { GalleryPageComponent } from './pages/gallery-page.component';

export const routes: Routes = [
  { path: '', component: StyleTransferPageComponent },
  { path: 'gallery', component: GalleryPageComponent },
  { path: '**', redirectTo: '' }
];
