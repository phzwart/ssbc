#!/bin/bash
# Deploy SSBC MCP Server to Google Cloud Run

set -e

PROJECT_ID=${1:-"your-gcp-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="ssbc-mcp"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       Deploying SSBC MCP Server to Google Cloud Run          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/install"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Build and deploy
echo "📦 Building and deploying..."
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --max-instances 10 \
  --platform managed

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                   Deployment Complete! ✅                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Test with:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "Use in Claude Desktop:"
echo "  Add to claude_desktop_config.json:"
echo "  {"
echo "    \"mcpServers\": {"
echo "      \"ssbc\": {"
echo "        \"url\": \"$SERVICE_URL\""
echo "      }"
echo "    }"
echo "  }"
echo ""
echo "Or register on OpenMCP: https://open-mcp.org"
