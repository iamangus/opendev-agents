package mcpclient

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/client"
	mcptransport "github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// DiscoveredTool represents a tool discovered from an external MCP server.
type DiscoveredTool struct {
	// ServerName is the name of the MCP server this tool belongs to.
	ServerName string
	// Tool is the MCP tool metadata.
	Tool mcp.Tool
}

// QualifiedName returns the namespaced name: "server.tool".
func (dt *DiscoveredTool) QualifiedName() string {
	return dt.ServerName + "." + dt.Tool.Name
}

// InputSchemaJSON returns the tool's input schema as json.RawMessage.
func (dt *DiscoveredTool) InputSchemaJSON() json.RawMessage {
	data, err := json.Marshal(dt.Tool.InputSchema)
	if err != nil {
		return json.RawMessage(`{"type":"object"}`)
	}
	return data
}

// Transport type constants.
const (
	TransportSSE            = "sse"
	TransportStreamableHTTP = "streamable-http"
)

// ServerConfig holds the configuration for connecting to an external MCP server.
type ServerConfig struct {
	Name      string            `yaml:"name" json:"name"`
	URL       string            `yaml:"url" json:"url"`
	Transport string            `yaml:"transport" json:"transport"` // "sse" (default) or "streamable-http"
	Headers   map[string]string `yaml:"headers" json:"headers"`
}

// connection holds a live MCP client connection and its discovered tools.
type connection struct {
	client *client.Client
	config ServerConfig
	tools  []mcp.Tool
}

// Pool manages connections to external MCP servers and provides
// tool discovery and proxied tool invocation.
type Pool struct {
	mu        sync.RWMutex
	conns     map[string]*connection // server name -> persistent connection
	ephemeral map[string]*connection // server name -> ephemeral connection (shadows conns)

	// onChange is called whenever the tool list changes (from any server).
	onChange func()
}

// NewPool creates a new MCP client pool.
func NewPool() *Pool {
	return &Pool{
		conns:     make(map[string]*connection),
		ephemeral: make(map[string]*connection),
	}
}

// OnToolsChanged registers a callback that fires when any server's tool list changes.
func (p *Pool) OnToolsChanged(fn func()) {
	p.onChange = fn
}

// Connect establishes connections to all configured MCP servers,
// initializes sessions, and discovers tools.
func (p *Pool) Connect(ctx context.Context, servers []ServerConfig) error {
	for _, srv := range servers {
		if err := p.connectOne(ctx, srv); err != nil {
			slog.Error("failed to connect to MCP server", "name", srv.Name, "url", srv.URL, "error", err)
			// Continue connecting to other servers; don't fail hard.
			continue
		}
	}
	return nil
}

// connectOne connects to a single MCP server.
func (p *Pool) connectOne(ctx context.Context, srv ServerConfig) error {
	transport := srv.Transport
	if transport == "" {
		transport = TransportSSE
	}
	slog.Info("connecting to MCP server", "name", srv.Name, "url", srv.URL, "transport", transport)

	var c *client.Client
	var err error

	switch transport {
	case TransportSSE:
		var opts []mcptransport.ClientOption
		if len(srv.Headers) > 0 {
			opts = append(opts, client.WithHeaders(srv.Headers))
		}
		c, err = client.NewSSEMCPClient(srv.URL, opts...)
	case TransportStreamableHTTP:
		var opts []mcptransport.StreamableHTTPCOption
		if len(srv.Headers) > 0 {
			opts = append(opts, mcptransport.WithHTTPHeaders(srv.Headers))
		}
		c, err = client.NewStreamableHttpClient(srv.URL, opts...)
	default:
		return fmt.Errorf("unknown transport %q for server %s (use 'sse' or 'streamable-http')", transport, srv.Name)
	}
	if err != nil {
		return fmt.Errorf("create %s client for %s: %w", transport, srv.Name, err)
	}

	conn := &connection{
		client: c,
		config: srv,
	}

	// Register notification handler before Start so we don't miss any.
	serverName := srv.Name
	c.OnNotification(func(notification mcp.JSONRPCNotification) {
		if notification.Method == mcp.MethodNotificationToolsListChanged {
			slog.Info("tool list changed notification received", "server", serverName)
			p.refreshTools(serverName)
		}
	})

	// Start the transport.
	startCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	if err := c.Start(startCtx); err != nil {
		c.Close()
		return fmt.Errorf("start %s client for %s: %w", transport, srv.Name, err)
	}

	// Initialize the MCP session.
	_, err = c.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo: mcp.Implementation{
				Name:    "agentfoundry",
				Version: "0.1.0",
			},
			Capabilities: mcp.ClientCapabilities{},
		},
	})
	if err != nil {
		c.Close()
		return fmt.Errorf("initialize MCP session for %s: %w", srv.Name, err)
	}

	// Discover tools.
	toolsResult, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		c.Close()
		return fmt.Errorf("list tools from %s: %w", srv.Name, err)
	}

	conn.tools = toolsResult.Tools

	p.mu.Lock()
	p.conns[srv.Name] = conn
	p.mu.Unlock()

	toolNames := make([]string, len(conn.tools))
	for i, t := range conn.tools {
		toolNames[i] = t.Name
	}
	slog.Info("connected to MCP server", "name", srv.Name, "tools", toolNames)

	return nil
}

// refreshTools re-fetches the tool list from a specific server.
func (p *Pool) refreshTools(serverName string) {
	p.mu.RLock()
	conn, ok := p.conns[serverName]
	p.mu.RUnlock()
	if !ok {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	toolsResult, err := conn.client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		slog.Error("failed to refresh tools", "server", serverName, "error", err)
		return
	}

	p.mu.Lock()
	conn.tools = toolsResult.Tools
	p.mu.Unlock()

	toolNames := make([]string, len(toolsResult.Tools))
	for i, t := range toolsResult.Tools {
		toolNames[i] = t.Name
	}
	slog.Info("refreshed tools from server", "server", serverName, "tools", toolNames)

	if p.onChange != nil {
		p.onChange()
	}
}

func (p *Pool) getConnection(name string) (*connection, bool) {
	if conn, ok := p.ephemeral[name]; ok {
		return conn, true
	}
	conn, ok := p.conns[name]
	return conn, ok
}

func (p *Pool) allConnections() map[string]*connection {
	merged := make(map[string]*connection, len(p.conns)+len(p.ephemeral))
	for k, v := range p.conns {
		merged[k] = v
	}
	for k, v := range p.ephemeral {
		merged[k] = v
	}
	return merged
}

// ListAllTools returns all discovered tools across all connected servers.
// Ephemeral servers shadow persistent servers with the same name.
func (p *Pool) ListAllTools() []DiscoveredTool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var all []DiscoveredTool
	for name, conn := range p.allConnections() {
		for _, t := range conn.tools {
			all = append(all, DiscoveredTool{
				ServerName: name,
				Tool:       t,
			})
		}
	}
	return all
}

// ListServerTools returns the tools from a specific server.
// Checks ephemeral first, then persistent.
func (p *Pool) ListServerTools(serverName string) ([]DiscoveredTool, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	conn, ok := p.getConnection(serverName)
	if !ok {
		return nil, false
	}

	tools := make([]DiscoveredTool, len(conn.tools))
	for i, t := range conn.tools {
		tools[i] = DiscoveredTool{
			ServerName: serverName,
			Tool:       t,
		}
	}
	return tools, true
}

// GetTool looks up a tool by its qualified name ("server.tool").
// Checks ephemeral first, then persistent.
func (p *Pool) GetTool(serverName, toolName string) (*DiscoveredTool, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	conn, ok := p.getConnection(serverName)
	if !ok {
		return nil, false
	}

	for _, t := range conn.tools {
		if t.Name == toolName {
			return &DiscoveredTool{
				ServerName: serverName,
				Tool:       t,
			}, true
		}
	}
	return nil, false
}

// CallTool invokes a tool on the appropriate external MCP server.
// Checks ephemeral first, then persistent.
func (p *Pool) CallTool(ctx context.Context, serverName, toolName string, arguments map[string]any) (*mcp.CallToolResult, error) {
	p.mu.RLock()
	conn, ok := p.getConnection(serverName)
	p.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("unknown MCP server: %s", serverName)
	}

	result, err := conn.client.CallTool(ctx, mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      toolName,
			Arguments: arguments,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("call tool %s.%s: %w", serverName, toolName, err)
	}

	return result, nil
}

// ListServerNames returns the names of all connected servers (persistent + ephemeral).
func (p *Pool) ListServerNames() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	merged := p.allConnections()
	names := make([]string, 0, len(merged))
	for name := range merged {
		names = append(names, name)
	}
	return names
}

// EphemeralConn is a short-lived connection to a single MCP server, intended
// for use within a single agent run. It is not registered in the global Pool
// and must be closed by the caller when the run completes.
type EphemeralConn struct {
	config ServerConfig
	client *client.Client
	tools  []mcp.Tool
}

// ConnectEphemeral connects to a single MCP server outside of the global pool
// and returns an EphemeralConn. The caller is responsible for calling Close
// when the connection is no longer needed.
func ConnectEphemeral(ctx context.Context, srv ServerConfig) (*EphemeralConn, error) {
	transport := srv.Transport
	if transport == "" {
		transport = TransportSSE
	}

	var c *client.Client
	var err error

	switch transport {
	case TransportSSE:
		var opts []mcptransport.ClientOption
		if len(srv.Headers) > 0 {
			opts = append(opts, client.WithHeaders(srv.Headers))
		}
		c, err = client.NewSSEMCPClient(srv.URL, opts...)
	case TransportStreamableHTTP:
		var opts []mcptransport.StreamableHTTPCOption
		if len(srv.Headers) > 0 {
			opts = append(opts, mcptransport.WithHTTPHeaders(srv.Headers))
		}
		c, err = client.NewStreamableHttpClient(srv.URL, opts...)
	default:
		return nil, fmt.Errorf("unknown transport %q for server %s (use 'sse' or 'streamable-http')", transport, srv.Name)
	}
	if err != nil {
		return nil, fmt.Errorf("create %s client for %s: %w", transport, srv.Name, err)
	}

	startCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	if err := c.Start(startCtx); err != nil {
		c.Close()
		return nil, fmt.Errorf("start %s client for %s: %w", transport, srv.Name, err)
	}

	_, err = c.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo: mcp.Implementation{
				Name:    "agentfoundry",
				Version: "0.1.0",
			},
			Capabilities: mcp.ClientCapabilities{},
		},
	})
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("initialize MCP session for %s: %w", srv.Name, err)
	}

	toolsResult, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("list tools from %s: %w", srv.Name, err)
	}

	toolNames := make([]string, len(toolsResult.Tools))
	for i, t := range toolsResult.Tools {
		toolNames[i] = t.Name
	}
	slog.Info("ephemeral MCP connection established", "name", srv.Name, "tools", toolNames)

	return &EphemeralConn{
		config: srv,
		client: c,
		tools:  toolsResult.Tools,
	}, nil
}

// ServerName returns the name this server was registered under.
func (e *EphemeralConn) ServerName() string {
	return e.config.Name
}

// ListTools returns all tools discovered from this ephemeral server.
func (e *EphemeralConn) ListTools() []DiscoveredTool {
	tools := make([]DiscoveredTool, len(e.tools))
	for i, t := range e.tools {
		tools[i] = DiscoveredTool{
			ServerName: e.config.Name,
			Tool:       t,
		}
	}
	return tools
}

// CallTool invokes a tool on this ephemeral server.
func (e *EphemeralConn) CallTool(ctx context.Context, toolName string, arguments map[string]any) (*mcp.CallToolResult, error) {
	return e.client.CallTool(ctx, mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      toolName,
			Arguments: arguments,
		},
	})
}

// Close shuts down the ephemeral MCP connection.
func (e *EphemeralConn) Close() {
	slog.Info("closing ephemeral MCP connection", "server", e.config.Name)
	e.client.Close()
}

// RegisterEphemeral adds an EphemeralConn to the pool under its server name.
// Tools from the ephemeral connection become visible via ListAllTools and
// CallTool. Call UnregisterEphemeral to remove it.
func (p *Pool) RegisterEphemeral(e *EphemeralConn) {
	conn := &connection{
		client: e.client,
		config: e.config,
		tools:  e.tools,
	}
	p.mu.Lock()
	p.ephemeral[e.config.Name] = conn
	p.mu.Unlock()
	slog.Info("registered ephemeral MCP server in pool", "name", e.config.Name, "tools", len(e.tools))
}

// UnregisterEphemeral removes a server from the pool by name and closes its
// underlying connection. If the server was not registered this is a no-op.
func (p *Pool) UnregisterEphemeral(name string) {
	p.mu.Lock()
	conn, ok := p.ephemeral[name]
	if !ok {
		p.mu.Unlock()
		return
	}
	delete(p.ephemeral, name)
	p.mu.Unlock()
	conn.client.Close()
	slog.Info("unregistered ephemeral MCP server from pool", "name", name)
}

// Close shuts down all MCP client connections.
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for name, conn := range p.ephemeral {
		slog.Info("closing ephemeral MCP connection", "server", name)
		conn.client.Close()
	}
	for name, conn := range p.conns {
		slog.Info("closing MCP client connection", "server", name)
		conn.client.Close()
	}
	p.ephemeral = make(map[string]*connection)
	p.conns = make(map[string]*connection)
}
