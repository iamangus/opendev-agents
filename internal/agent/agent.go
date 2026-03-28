package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/angoo/agentfile/internal/agentlog"
	"github.com/angoo/agentfile/internal/config"
	"github.com/angoo/agentfile/internal/llm"
	"github.com/angoo/agentfile/internal/mcpclient"
)

// AgentResolver resolves agent names to definitions.
type AgentResolver interface {
	GetAgentDef(name string) (*config.Definition, bool)
}

// Reporter receives status updates during an agent run.
// Implementations must be safe for concurrent use.
// A nil Reporter is valid and silently drops all events.
type Reporter interface {
	Update(status string)
}

// HistoryRecorder captures detailed execution history during an agent run.
// Implementations must be safe for concurrent use.
// A nil HistoryRecorder is valid and silently drops all events.
type HistoryRecorder interface {
	StartTurn(turnNum int)
	RecordRequest(requestJSON string)
	RecordResponse(responseJSON string)
	EndTurn()
	StartToolCall(toolCallID, toolName, arguments string)
	EndToolCall(result string, status ToolCallStatus, errMsg string)
}

// ToolCallStatus represents the outcome of a tool call.
type ToolCallStatus string

const (
	ToolCallStatusSuccess ToolCallStatus = "success"
	ToolCallStatusError   ToolCallStatus = "error"
)

// Runtime manages agent execution.
type Runtime struct {
	resolver  AgentResolver
	pool      *mcpclient.Pool
	llmClient llm.Client
	logger    *agentlog.Logger // optional; nil disables run logging
}

// NewRuntime creates a new agent runtime.
func NewRuntime(resolver AgentResolver, pool *mcpclient.Pool, llmClient llm.Client) *Runtime {
	return &Runtime{
		resolver:  resolver,
		pool:      pool,
		llmClient: llmClient,
	}
}

// SetLogger attaches a run logger to the runtime. Must be called before any
// runs start; it is not safe to call concurrently with Run/RunWithReporter.
func (rt *Runtime) SetLogger(l *agentlog.Logger) {
	rt.logger = l
}

// GetToolNames returns the list of tool names that an agent will have access to.
func (rt *Runtime) GetToolNames(def *config.Definition, ephemeral []*mcpclient.EphemeralConn) []string {
	toolDefs, _ := rt.buildToolSet(def, ephemeral)
	names := make([]string, len(toolDefs))
	for i, td := range toolDefs {
		names[i] = td.Function.Name
	}
	return names
}

// report sends a status update if r is non-nil.
func report(r Reporter, status string) {
	if r != nil {
		r.Update(status)
	}
}

// Run executes an agent with the given user input and returns the final text response.
func (rt *Runtime) Run(ctx context.Context, def *config.Definition, userInput string, ephemeral ...*mcpclient.EphemeralConn) (string, error) {
	result, _, err := rt.RunWithReporter(ctx, def, userInput, nil, nil, ephemeral...)
	return result, err
}

// RunWithReporter is like Run but emits status updates via r throughout execution.
// history contains any prior conversation turns (from previous user/assistant exchanges)
// and is prepended between the system prompt and the new user message.
// Any ephemeral MCP connections provided will have their tools appended to the
// agent's static tool list for the duration of this run only.
// It returns the final text response, the full updated message history, and any error.
func (rt *Runtime) RunWithReporter(ctx context.Context, def *config.Definition, userInput string, r Reporter, history []llm.Message, ephemeral ...*mcpclient.EphemeralConn) (string, []llm.Message, error) {
	return rt.RunWithHistory(ctx, def, userInput, r, nil, history, ephemeral...)
}

// RunWithHistory is like RunWithReporter but also records detailed execution history.
func (rt *Runtime) RunWithHistory(ctx context.Context, def *config.Definition, userInput string, r Reporter, hr HistoryRecorder, history []llm.Message, ephemeral ...*mcpclient.EphemeralConn) (string, []llm.Message, error) {
	report(r, "Thinking…")

	// Build tool definitions for the LLM
	toolDefs, toolMap := rt.buildToolSet(def, ephemeral)

	toolNames := make([]string, len(toolDefs))
	for i, td := range toolDefs {
		toolNames[i] = td.Function.Name
	}
	slog.Info("agent run started", "agent", def.Name, "tools", toolNames, "input_len", len(userInput), "history_len", len(history))

	// Start run log
	model := def.Model
	var rl *agentlog.RunLog
	if rt.logger != nil {
		rl = rt.logger.ForRun(def.Name, model)
		rl.Start(userInput)
		if len(history) > 0 {
			rl.HistorySummary(len(history))
		}
	}

	// Initialize conversation: system prompt + prior history + new user message
	messages := make([]llm.Message, 0, 2+len(history))
	messages = append(messages, llm.Message{Role: "system", Content: def.SystemPrompt})
	messages = append(messages, history...)
	messages = append(messages, llm.Message{Role: "user", Content: userInput})

	maxTurns := def.MaxTurns
	if maxTurns == 0 {
		maxTurns = 10
	}

	for turn := 0; turn < maxTurns; turn++ {
		slog.Debug("agent turn", "agent", def.Name, "turn", turn)
		if rl != nil {
			rl.Turn(turn + 1)
		}
		if hr != nil {
			hr.StartTurn(turn + 1)
		}

		req := &llm.ChatRequest{
			Model:    def.Model,
			Messages: messages,
		}
		if len(toolDefs) > 0 {
			req.Tools = toolDefs
		}
		if def.ForceJSON {
			req.ResponseFormat = &llm.ResponseFormat{Type: "json_object"}
		}

		// Log the exact JSON being sent to the LLM
		var reqJSON []byte
		if rl != nil || hr != nil {
			var err error
			reqJSON, err = json.MarshalIndent(req, "", "  ")
			if err == nil {
				if rl != nil {
					rl.Request(reqJSON)
				}
				if hr != nil {
					hr.RecordRequest(string(reqJSON))
				}
			}
		}

		resp, err := rt.llmClient.ChatCompletion(ctx, req)
		if err != nil {
			runErr := fmt.Errorf("llm call failed on turn %d: %w", turn, err)
			if rl != nil {
				rl.Failed(runErr)
			}
			return "", messages, runErr
		}

		if len(resp.Choices) == 0 {
			runErr := fmt.Errorf("no choices in LLM response on turn %d", turn)
			if rl != nil {
				rl.Failed(runErr)
			}
			return "", messages, runErr
		}

		// Log the exact JSON received from the LLM
		if rl != nil || hr != nil {
			if respJSON, err := json.MarshalIndent(resp, "", "  "); err == nil {
				if rl != nil {
					rl.Response(respJSON)
				}
				if hr != nil {
					hr.RecordResponse(string(respJSON))
				}
			}
		}

		// Merge all choices with index 0 into a single message.
		// Some providers (e.g. claude-haiku via certain proxies) split the
		// text content and tool_calls across two separate choice objects that
		// both carry "index": 0, which is non-standard. Merging them here
		// ensures tool_calls are never silently dropped.
		var assistantMsg llm.Message
		for _, c := range resp.Choices {
			if c.Index != 0 {
				continue
			}
			if assistantMsg.Role == "" {
				assistantMsg.Role = c.Message.Role
			}
			if assistantMsg.Content == nil {
				assistantMsg.Content = c.Message.Content
			}
			assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, c.Message.ToolCalls...)
		}

		// Add assistant message to conversation
		messages = append(messages, assistantMsg)

		// If no tool calls, we're done
		if len(assistantMsg.ToolCalls) == 0 {
			content, _ := assistantMsg.Content.(string)
			slog.Info("agent run completed", "agent", def.Name, "turns", turn+1)
			if rl != nil {
				rl.Completed(turn + 1)
			}
			if hr != nil {
				hr.EndTurn()
			}
			// Return history excluding the system prompt so it can be replayed next turn.
			// Slice off the leading system message: messages[1:] = history + new user msg + assistant reply.
			return content, messages[1:], nil
		}

		// Process tool calls in parallel, bounded by MaxConcurrentTools.
		// 0 (default) = unlimited; 1 = serial; N > 1 = capped at N.
		type toolResult struct {
			toolCallID string
			content    string
			err        error
		}

		results := make([]toolResult, len(assistantMsg.ToolCalls))

		var sem chan struct{}
		if def.MaxConcurrentTools == 1 {
			sem = make(chan struct{}, 1)
		} else if def.MaxConcurrentTools > 1 {
			sem = make(chan struct{}, def.MaxConcurrentTools)
		}

		var wg sync.WaitGroup
		for i, tc := range assistantMsg.ToolCalls {
			wg.Add(1)
			go func(i int, tc llm.ToolCall) {
				defer wg.Done()
				if sem != nil {
					sem <- struct{}{}
					defer func() { <-sem }()
				}
				report(r, toolStatus(tc.Function.Name, toolMap))

				// Record tool call start
				if hr != nil {
					hr.StartToolCall(tc.ID, tc.Function.Name, tc.Function.Arguments)
				}

				result, err := rt.executeTool(ctx, tc, toolMap, r)

				// Record tool call end
				if hr != nil {
					var status ToolCallStatus = ToolCallStatusSuccess
					var errMsg string
					if err != nil {
						status = ToolCallStatusError
						errMsg = err.Error()
					}
					hr.EndToolCall(result, status, errMsg)
				}

				if err != nil {
					slog.Warn("tool call failed", "agent", def.Name, "tool", tc.Function.Name, "error", err)
					results[i] = toolResult{toolCallID: tc.ID, content: fmt.Sprintf("Error: %s", err.Error()), err: err}
				} else {
					results[i] = toolResult{toolCallID: tc.ID, content: result}
				}
			}(i, tc)
		}
		wg.Wait()

		for _, res := range results {
			messages = append(messages, llm.Message{
				Role:       "tool",
				Content:    res.content,
				ToolCallID: res.toolCallID,
			})
		}

		if hr != nil {
			hr.EndTurn()
		}
		report(r, "Thinking…")
	}

	runErr := fmt.Errorf("agent %s exceeded max turns (%d)", def.Name, maxTurns)
	if rl != nil {
		rl.Failed(runErr)
	}
	return "", messages[1:], runErr
}

// toolStatus returns a human-readable status string for a tool call.
func toolStatus(llmName string, toolMap map[string]*toolRef) string {
	ref, ok := toolMap[llmName]
	if !ok {
		return fmt.Sprintf("Calling %s…", llmName)
	}
	if ref.agentDef != nil {
		return fmt.Sprintf("Asking %s…", ref.agentDef.Name)
	}
	// Pretty-print the tool name: "searxng_web_search" → "Running web search…"
	switch ref.toolName {
	case "searxng_web_search", "web_search":
		return "Running web search…"
	case "web_url_read", "url_read", "fetch_url":
		return "Reading webpage…"
	default:
		name := strings.ReplaceAll(ref.toolName, "_", " ")
		return fmt.Sprintf("Using %s…", name)
	}
}

// toolRef describes how to call a tool: either an MCP tool or a sub-agent.
type toolRef struct {
	// For MCP tools from the global pool (namespaced: "server.tool")
	serverName string
	toolName   string

	// For MCP tools from an ephemeral connection
	ephemeral *mcpclient.EphemeralConn

	// For agent-as-tool
	agentDef *config.Definition
}

// buildToolSet creates the LLM tool definitions and a lookup map for an agent.
// Static tools come from def.Tools (resolved via the global pool or agent registry).
// Ephemeral tools are appended from any provided EphemeralConn instances.
func (rt *Runtime) buildToolSet(def *config.Definition, ephemeral []*mcpclient.EphemeralConn) ([]llm.ToolDef, map[string]*toolRef) {
	toolDefs := make([]llm.ToolDef, 0, len(def.Tools))
	toolMap := make(map[string]*toolRef)

	// --- Static tools from the agent definition ---
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
			// Dots aren't valid in OpenAI function names, so use double underscore.
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

	// --- Dynamic tools from ephemeral connections ---
	for _, conn := range ephemeral {
		for _, dt := range conn.ListTools() {
			llmName := dt.ServerName + "__" + dt.Tool.Name
			toolDefs = append(toolDefs, llm.ToolDef{
				Type: "function",
				Function: llm.FunctionDef{
					Name:        llmName,
					Description: dt.Tool.Description,
					Parameters:  dt.InputSchemaJSON(),
				},
			})
			toolMap[llmName] = &toolRef{
				serverName: dt.ServerName,
				toolName:   dt.Tool.Name,
				ephemeral:  conn,
			}
		}
		slog.Info("appended ephemeral tools to agent tool set",
			"agent", def.Name, "server", conn.ServerName(), "count", len(conn.ListTools()))
	}

	return toolDefs, toolMap
}

// executeTool runs a tool call and returns the text result.
func (rt *Runtime) executeTool(ctx context.Context, tc llm.ToolCall, toolMap map[string]*toolRef, r Reporter) (string, error) {
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
		// Sub-agents do not inherit ephemeral connections or history from the parent run.
		result, _, err := rt.RunWithReporter(ctx, ref.agentDef, params.Message, r, nil)
		return result, err
	}

	// MCP tool call
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		return "", fmt.Errorf("parse tool arguments: %w", err)
	}

	slog.Info("tool call: mcp", "server", ref.serverName, "tool", ref.toolName, "args", args)

	var result *mcp.CallToolResult
	var err error
	if ref.ephemeral != nil {
		result, err = ref.ephemeral.CallTool(ctx, ref.toolName, args)
	} else {
		result, err = rt.pool.CallTool(ctx, ref.serverName, ref.toolName, args)
	}
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
