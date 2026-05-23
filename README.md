# AI Image Classifier

Upload any image and get AI-powered classification labels,
confidence scores, and tags using Claude (Anthropic).

## Stack
- **Frontend**: React + Vite + TailwindCSS
- **Backend**: Node.js + Express + Anthropic SDK

## Setup

### Backend
cd backend
cp .env.example .env      # Add your ANTHROPIC_API_KEY
npm install
npm run dev

### Frontend
cd frontend
cp .env.example .env      # Set VITE_API_BASE_URL
npm install
npm run dev

## Environment Variables
Never commit .env files. Use .env.example as a template.
