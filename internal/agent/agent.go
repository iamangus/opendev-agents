package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/angoo/agentfile/internal/config"
	"github.com/angoo/agentfile/internal/llm"
	"github.com/angoo/agentfile/internal/mcpclient"
)

// AgentResolver resolves agent names to definitions.
type AgentResolver interface {
	GetAgentDef(name string) (*config.Definition, bool)
}

// Runtime manages agent execution.
type Runtime struct {
	resolver  AgentResolver
	pool      *mcpclient.Pool
	llmClient llm.Client
}

// NewRuntime creates a new agent runtime.
func NewRuntime(resolver AgentResolver, pool *mcpclient.Pool, llmClient llm.Client) *Runtime {
	return &Runtime{
		resolver:  resolver,
		pool:      pool,
		llmClient: llmClient,
	}
}

// Run executes an agent with the given user input and returns the final text response.
func (rt *Runtime) Run(ctx context.Context, def *config.Definition, userInput string) (string, error) {
	slog.Info("agent run started", "agent", def.Name, "input_len", len(userInput))

	// Build tool definitions for the LLM
	toolDefs, toolMap := rt.buildToolSet(def)

	// Initialize conversation
	messages := []llm.Message{
		{Role: "system", Content: def.SystemPrompt},
		{Role: "user", Content: userInput},
	}

	maxTurns := def.MaxTurns
	if maxTurns == 0 {
		maxTurns = 10
	}

	for turn := 0; turn < maxTurns; turn++ {
		slog.Debug("agent turn", "agent", def.Name, "turn", turn)

		req := &llm.ChatRequest{
			Model:    def.Model,
			Messages: messages,
		}
		if len(toolDefs) > 0 {
			req.Tools = toolDefs
		}

		resp, err := rt.llmClient.ChatCompletion(ctx, req)
		if err != nil {
			return "", fmt.Errorf("llm call failed on turn %d: %w", turn, err)
		}

		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no choices in LLM response on turn %d", turn)
		}

		choice := resp.Choices[0]
		assistantMsg := choice.Message

		// Add assistant message to conversation
		messages = append(messages, assistantMsg)

		// If no tool calls, we're done
		if len(assistantMsg.ToolCalls) == 0 {
			content, _ := assistantMsg.Content.(string)
			slog.Info("agent run completed", "agent", def.Name, "turns", turn+1)
			return content, nil
		}

		// Process tool calls
		for _, tc := range assistantMsg.ToolCalls {
			result, err := rt.executeTool(ctx, tc, toolMap)

			var resultContent string
			if err != nil {
				resultContent = fmt.Sprintf("Error: %s", err.Error())
				slog.Warn("tool call failed", "agent", def.Name, "tool", tc.Function.Name, "error", err)
			} else {
				resultContent = result
			}

			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    resultContent,
				ToolCallID: tc.ID,
			})
		}
	}

	return "", fmt.Errorf("agent %s exceeded max turns (%d)", def.Name, maxTurns)
}

// toolRef describes how to call a tool: either an MCP tool or a sub-agent.
type toolRef struct {
	// For MCP tools (namespaced: "server.tool")
	serverName string
	toolName   string

	// For agent-as-tool
	agentDef *config.Definition
}

// buildToolSet creates the LLM tool definitions and a lookup map for an agent.
func (rt *Runtime) buildToolSet(def *config.Definition) ([]llm.ToolDef, map[string]*toolRef) {
	toolDefs := make([]llm.ToolDef, 0, len(def.Tools))
	toolMap := make(map[string]*toolRef)

	for _, ref := range def.Tools {
		// Check if it's a namespaced MCP tool: "server.tool"
		if serverName, toolName, ok := parseToolRef(ref); ok {
			dt, found := rt.pool.GetTool(serverName, toolName)
			if !found {
				slog.Warn("agent references unknown MCP tool, skipping",
					"agent", def.Name, "ref", ref)
				continue
			}

			// Use the qualified name as the function name for the LLM.
			// Dots aren't valid in OpenAI function names, so use underscore.
			llmName := serverName + "__" + toolName

			toolDefs = append(toolDefs, llm.ToolDef{
				Type: "function",
				Function: llm.FunctionDef{
					Name:        llmName,
					Description: dt.Tool.Description,
					Parameters:  dt.InputSchemaJSON(),
				},
			})
			toolMap[llmName] = &toolRef{serverName: serverName, toolName: toolName}
			continue
		}

		// Otherwise, check if it's an agent name
		if agentDef, ok := rt.resolver.GetAgentDef(ref); ok {
			schema := json.RawMessage(`{
				"type": "object",
				"properties": {
					"message": {
						"type": "string",
						"description": "The message/request to send to this agent"
					}
				},
				"required": ["message"]
			}`)
			toolDefs = append(toolDefs, llm.ToolDef{
				Type: "function",
				Function: llm.FunctionDef{
					Name:        ref,
					Description: agentDef.Description,
					Parameters:  schema,
				},
			})
			toolMap[ref] = &toolRef{agentDef: agentDef}
			continue
		}

		slog.Warn("agent references unresolvable tool/agent, skipping",
			"agent", def.Name, "ref", ref)
	}

	return toolDefs, toolMap
}

// executeTool runs a tool call and returns the text result.
func (rt *Runtime) executeTool(ctx context.Context, tc llm.ToolCall, toolMap map[string]*toolRef) (string, error) {
	ref, ok := toolMap[tc.Function.Name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", tc.Function.Name)
	}

	// Agent-as-tool
	if ref.agentDef != nil {
		var params struct {
			Message string `json:"message"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &params); err != nil {
			return "", fmt.Errorf("parse agent call input: %w", err)
		}
		slog.Info("tool call: agent", "agent", ref.agentDef.Name, "input_len", len(params.Message))
		return rt.Run(ctx, ref.agentDef, params.Message)
	}

	// MCP tool call
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("parse tool arguments: %w", err)
	}

	slog.Info("tool call: mcp", "server", ref.serverName, "tool", ref.toolName, "args", args)

	result, err := rt.pool.CallTool(ctx, ref.serverName, ref.toolName, args)
	if err != nil {
		return "", err
	}

	if result.IsError {
		return "", fmt.Errorf("tool returned error: %s", extractText(result))
	}

	slog.Info("tool call: mcp completed", "server", ref.serverName, "tool", ref.toolName, "result_len", len(extractText(result)))
	return extractText(result), nil
}

// parseToolRef splits "server.tool" into its parts.
// Returns false if the ref doesn't contain a dot (i.e., it's an agent name).
func parseToolRef(ref string) (serverName, toolName string, ok bool) {
	idx := strings.Index(ref, ".")
	if idx < 0 {
		return "", "", false
	}
	return ref[:idx], ref[idx+1:], true
}

// extractText pulls the text content from an MCP CallToolResult.
func extractText(result *mcp.CallToolResult) string {
	var parts []string
	for _, c := range result.Content {
		switch v := c.(type) {
		case mcp.TextContent:
			parts = append(parts, v.Text)
		}
	}
	return strings.Join(parts, "\n")
}
