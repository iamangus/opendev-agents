# Structured Outputs (JSON Schema) Design

**Date:** 2026-03-28
**Status:** Approved

## Overview

Add proper JSON structured outputs (`response_format: json_schema`) support to agent definitions, the web UI, and the run API. Builds on the existing `force_json` (json_object) mechanism — when a `structured_output` block is present on a definition it takes precedence; `force_json: true` without a schema continues to work as before.

## Data Model

### New type: `StructuredOutput` (`internal/config/definition.go`)

```go
type StructuredOutput struct {
    Name   string          `yaml:"name"   json:"name"`
    Schema json.RawMessage `yaml:"schema" json:"schema"`
    Strict bool            `yaml:"strict" json:"strict"`
}
```

Mirrors the OpenAI `json_schema` block exactly so it can be passed through without transformation.

### `Definition` gains a new field

```go
StructuredOutput *StructuredOutput `yaml:"structured_output,omitempty" json:"structured_output,omitempty"`
```

### LLM client (`internal/llm/client.go`)

`ResponseFormat` gains a `JSONSchema` field:

```go
type ResponseFormat struct {
    Type       string      `json:"type"`
    JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

type JSONSchema struct {
    Name   string          `json:"name"`
    Schema json.RawMessage `json:"schema"`
    Strict bool            `json:"strict"`
}
```

### YAML example

```yaml
structured_output:
  name: analysis_result
  strict: true
  schema:
    type: object
    properties:
      summary:
        type: string
      score:
        type: integer
    required: [summary, score]
    additionalProperties: false
```

## Agent Runtime (`internal/agent/agent.go`)

Priority logic when building `ResponseFormat` for each LLM request:

1. If an override `*StructuredOutput` was passed at run time → use it (`json_schema`)
2. Else if `def.StructuredOutput != nil` → use it (`json_schema`)
3. Else if `def.ForceJSON` → use `json_object`
4. Else → no response format

`RunWithHistory()` accepts an optional `*config.StructuredOutput` override parameter.

## Run API (`internal/api/http.go`)

`RunRequest` gains an optional field:

```go
type RunRequest struct {
    Message         string                    `json:"message"`
    History         []llm.Message             `json:"history,omitempty"`
    MCPServers      []mcpclient.ServerConfig   `json:"mcp_servers,omitempty"`
    ResponseSchema  *config.StructuredOutput  `json:"response_schema,omitempty"`
}
```

If `ResponseSchema` is non-nil in the request, it overrides the definition's `StructuredOutput` for that run only.

## Web UI (`internal/web/`)

On the agent edit page (`agents.html`), below the existing YAML editor, add:

- A **checkbox** — "Enable structured output (JSON Schema)"
- A **JSON textarea** — revealed when checked; pre-populated from the definition's `structured_output` block if present
- On save, the JSON is serialized back into the YAML under `structured_output:`

The JSON in the textarea is the full structured output object (`name`, `schema`, `strict`) matching the API format exactly — what you define in the UI is what you pass in the API.

## Behaviour Summary

| Definition has `structured_output` | API passes `response_schema` | Result |
|---|---|---|
| No | No | No response format (or `json_object` if `force_json: true`) |
| Yes | No | `json_schema` from definition |
| Yes | Yes | `json_schema` from API (override) |
| No | Yes | `json_schema` from API |

## Out of Scope

- Visual schema builder (users edit raw JSON)
- Per-field validation of the JSON Schema contents
- Provider compatibility warnings
