import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <nav class="bg-white shadow-md">
      <div class="container mx-auto px-4">
        <div class="flex justify-between items-center h-16">
          <div class="flex items-center">
            <h1 class="text-xl font-bold text-purple-900">Neural Style Transfer</h1>
          </div>
          <div class="flex space-x-4">
            <a routerLink="/"
               routerLinkActive="text-purple-600"
               [routerLinkActiveOptions]="{exact: true}"
               class="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium">
              Create
            </a>
            <a routerLink="/gallery"
               routerLinkActive="text-purple-600"
               class="text-gray-700 hover:text-purple-600 px-3 py-2 rounded-md text-sm font-medium">
              Gallery
            </a>
          </div>
        </div>
      </div>
    </nav>
    <router-outlet></router-outlet>
  `,
  styles: []
})
export class AppComponent {}
