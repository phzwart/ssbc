# SSBC MCP Server

Model Context Protocol (MCP) server exposing SSBC functionality as AI-callable tools.

## What is MCP?

[Model Context Protocol](https://modelcontextprotocol.io/) is Anthropic's standard for connecting AI assistants to tools and data sources. This server allows Claude and other MCP clients to compute Small-Sample Beta Corrections.

## Available Tools

### `compute_ssbc_correction`

Computes the SSBC-corrected miscoverage rate for finite-sample PAC guarantees.

**Parameters:**
- `alpha_target` (float): Target miscoverage rate (e.g., 0.10 for 90% coverage)
- `n` (int): Calibration set size
- `delta` (float): PAC risk parameter (e.g., 0.05 for 95% confidence)
- `mode` (str): "beta" (default) or "beta-binomial"

**Returns:**
```json
{
  "alpha_corrected": 0.0571,
  "u_star": 95,
  "pac_mass": 0.9549,
  "guarantee": "With 95.0% probability, coverage ≥ 90.0%",
  "explanation": "Use α'=0.0571 instead of α=0.10..."
}
```

## Local Usage

### 1. Install Dependencies

```bash
pip install -e ".[mcp]"
```

### 2. Run Server Locally

```bash
python -m ssbc.mcp_server
```

### 3. Test with MCP Inspector

```bash
# Install MCP inspector
npm install -g @modelcontextprotocol/inspector

# Test server
mcp-inspector python -m ssbc.mcp_server
```

## Deployment Options

### Google Cloud Run (Recommended)

```bash
# Build and deploy
gcloud run deploy ssbc-mcp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# Get URL
gcloud run services describe ssbc-mcp --region us-central1 --format 'value(status.url)'
```

### Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

### Docker (Self-Hosted)

```bash
# Build image
docker build -t ssbc-mcp .

# Run locally
docker run -p 8000:8000 ssbc-mcp

# Deploy to your server
docker push your-registry/ssbc-mcp
```

## Usage in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ssbc": {
      "command": "python",
      "args": ["-m", "ssbc.mcp_server"],
      "env": {}
    }
  }
}
```

Or for remote server:

```json
{
  "mcpServers": {
    "ssbc": {
      "url": "https://your-ssbc-mcp.run.app"
    }
  }
}
```

## Register on OpenMCP (Optional)

To make your server discoverable:

1. Go to https://open-mcp.org
2. Register your server URL
3. Add description and tags
4. Others can find and use your SSBC tool

## Example Usage (from Claude)

```
User: "I have 50 calibration points and want 90% coverage with 95% confidence.
       What should my corrected alpha be?"

Claude: [uses compute_ssbc_correction tool]
        "For n=50, α=0.10, δ=0.05, use α'=0.0571. This ensures ≥90% coverage
        with 95% probability."
```

## API Endpoint

Once deployed, you can also call it directly:

```bash
curl -X POST https://your-server.com/tools/compute_ssbc_correction \
  -H "Content-Type: application/json" \
  -d '{
    "alpha_target": 0.10,
    "n": 100,
    "delta": 0.05,
    "mode": "beta"
  }'
```

## Cost Estimates

**Google Cloud Run** (with free tier):
- First 2M requests/month: FREE
- After: ~$0.40 per 1M requests
- SSBC calls are lightweight (~1ms compute)
- Expected cost: $0-5/month for moderate usage

**Railway.app**:
- Free tier: $5 credit/month
- After: ~$10/month for always-on
- Good for development/testing

**Fly.io**:
- Free tier: 3 small VMs
- After: ~$2/month per VM
- Global edge deployment

## Security

For public deployment:
- Consider rate limiting (prevent abuse)
- Add authentication if needed (API keys)
- Monitor usage (Cloud Run provides free monitoring)

## License

MIT License - same as SSBC package
