#!/bin/bash
set -e

PROJECT_ID=perditio-website
REGION=us-central1
SERVICE_NAME=reporium-api
IMAGE=gcr.io/$PROJECT_ID/$SERVICE_NAME

gcloud builds submit --tag $IMAGE

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=$DATABASE_URL,INGESTION_API_KEY=$INGESTION_API_KEY,GH_USERNAME=perditioinc
