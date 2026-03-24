# ADR 005: MCP as the Agent-Facing Interface

- Status: Accepted

## Context

Reporium already has a REST API for application and automation use, but agent workflows benefit from a smaller, task-oriented interface instead of raw endpoint discovery. The suite now includes `reporium-mcp`, which wraps live API capabilities such as repo search, taxonomy browsing, portfolio questioning, and analytics. Agents are a first-class consumer of Reporium, not just a secondary integration.

## Decision

Reporium exposes an MCP server as the preferred agent-facing layer while keeping the REST API as the underlying system contract. MCP tools map to high-value portfolio actions such as semantic search, repo lookup, taxonomy exploration, portfolio insights, gap analysis, and quality lookup. The REST API remains the implementation backbone, but MCP becomes the interface optimized for AI-native reasoning and tool use.

## Consequences

Agents can query the portfolio in fewer steps and with less brittle prompt logic because the tool layer is curated around portfolio tasks rather than around raw transport concerns. This makes the system more AI-native: an agent can ask for gaps, trends, similar repos, or cross-dimension counts directly instead of composing multiple low-level requests. The tradeoff is that the MCP surface must be kept current as new API capabilities land. That creates a maintenance obligation, but it is acceptable because it gives the project a clearer separation between backend data contracts and agent ergonomics.
