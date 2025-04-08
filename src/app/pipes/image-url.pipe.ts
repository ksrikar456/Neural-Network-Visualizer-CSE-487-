import { Pipe, PipeTransform } from '@angular/core';
import { environment } from '../../environments/environment';

@Pipe({
  name: 'imageUrl',
  standalone: true
})
export class ImageUrlPipe implements PipeTransform {
  private baseUrl = environment.apiUrl.substring(0, environment.apiUrl.lastIndexOf('/api'));

  transform(url: string | null | undefined): string {
    if (!url) {
      return '';
    }

    // If the URL already starts with http, it's already absolute
    if (url.startsWith('http')) {
      return url;
    }

    // Otherwise, prepend the base URL
    return `${this.baseUrl}${url}`;
  }
} 