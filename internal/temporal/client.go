package temporal

import (
	"context"
	"crypto/rand"
	"fmt"
	"log/slog"

	"github.com/angoo/agentfile/internal/config"
	"github.com/angoo/agentfile/internal/llm"
	"github.com/angoo/agentfile/internal/mcpclient"

	"go.temporal.io/sdk/client"
)

const (
	TaskQueue    = "agent-temporal-worker"
	WorkflowType = "RunAgentWorkflow"
)

type RunAgentParams struct {
	AgentName      string                   `json:"agent_name"`
	Message        string                   `json:"message"`
	History        []llm.Message            `json:"history,omitempty"`
	MCPServers     []mcpclient.ServerConfig `json:"mcp_servers,omitempty"`
	ResponseSchema *config.StructuredOutput `json:"response_schema,omitempty"`
	StreamID       string                   `json:"stream_id,omitempty"`
}

type RunAgentResult struct {
	Response string        `json:"response"`
	History  []llm.Message `json:"history,omitempty"`
}

type Client struct {
	c client.Client
}

type Config struct {
	HostPort  string
	Namespace string
	APIKey    string
}

func NewClient(hostPort, namespace, apiKey string) (*Client, error) {
	opts := client.Options{
		HostPort:  hostPort,
		Namespace: namespace,
	}
	if apiKey != "" {
		opts.Credentials = client.NewAPIKeyStaticCredentials(apiKey)
	}

	c, err := client.Dial(opts)
	if err != nil {
		return nil, fmt.Errorf("dial temporal: %w", err)
	}
	slog.Info("connected to temporal server", "host", hostPort, "namespace", namespace)
	return &Client{c: c}, nil
}

func (c *Client) ExecuteWorkflow(ctx context.Context, params RunAgentParams) (string, error) {
	workflowID := params.AgentName + "-" + randomID()
	workflowOpts := client.StartWorkflowOptions{
		ID:        workflowID,
		TaskQueue: TaskQueue,
	}

	run, err := c.c.ExecuteWorkflow(ctx, workflowOpts, WorkflowType, params)
	if err != nil {
		return "", fmt.Errorf("start workflow: %w", err)
	}
	slog.Info("started temporal workflow", "workflow_id", workflowID, "agent", params.AgentName)

	return run.GetID(), nil
}

func (c *Client) ExecuteWorkflowSync(ctx context.Context, params RunAgentParams) (*RunAgentResult, error) {
	workflowID := params.AgentName + "-" + randomID()
	workflowOpts := client.StartWorkflowOptions{
		ID:        workflowID,
		TaskQueue: TaskQueue,
	}

	run, err := c.c.ExecuteWorkflow(ctx, workflowOpts, WorkflowType, params)
	if err != nil {
		return nil, fmt.Errorf("start workflow: %w", err)
	}
	slog.Info("started temporal workflow (sync)", "workflow_id", workflowID, "agent", params.AgentName)

	var result RunAgentResult
	if err := run.Get(ctx, &result); err != nil {
		return nil, fmt.Errorf("workflow execution: %w", err)
	}
	return &result, nil
}

func (c *Client) Close() {
	c.c.Close()
}

func randomID() string {
	var buf [8]byte
	rand.Read(buf[:])
	return fmt.Sprintf("%x", buf[:])
}
