#!/bin/bash

# Create a temporary deployment directory
mkdir deploy

# Copy backend files
cp -r backend/* deploy/

# Copy frontend files
mkdir -p deploy/frontend/static
mkdir -p deploy/frontend/templates
cp -r frontend/static/* deploy/frontend/static/
cp -r frontend/templates/* deploy/frontend/templates/

# Deploy to Google App Engine
gcloud app deploy deploy/app.yaml

# Clean up
rm -rf deploy

