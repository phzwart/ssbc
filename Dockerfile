# Dockerfile for SSBC MCP Server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md ./

# Install package with MCP dependencies
RUN pip install --no-cache-dir -e ".[mcp]"

# Expose port for MCP server
EXPOSE 8000

# Run MCP server
CMD ["python", "-m", "ssbc.mcp_server"]
