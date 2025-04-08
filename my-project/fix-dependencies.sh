#!/bin/bash
echo "Installing correct versions of TailwindCSS and related packages..."

# Remove existing packages that might cause conflicts
npm uninstall tailwindcss postcss autoprefixer

# Install correct versions
npm install -D tailwindcss@^3.3.0 postcss@^8.4.31 autoprefixer@^10.4.14

echo "Done! Now try running 'npm start' to start the application." 