package web

import (
	"context"
	"embed"
	"fmt"
	"html/template"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/angoo/agentfile/internal/agent"
	"github.com/angoo/agentfile/internal/config"
	"github.com/angoo/agentfile/internal/llm"
	"github.com/angoo/agentfile/internal/mcpclient"
	yamlpkg "gopkg.in/yaml.v3"
)

//go:embed templates/*.html
var templateFS embed.FS

// DefinitionStore is the subset of config.Loader used by the web handler.
type DefinitionStore interface {
	ListDefinitions() []*config.Definition
	GetRawDefinition(name string) ([]byte, error)
	SaveDefinition(def *config.Definition) error
	DeleteDefinition(name string) error
	GetDefinition(name string) *config.Definition
	SaveRawDefinition(name string, data []byte) error
}

// Message is a single turn in a chat.
type Message struct {
	Role    string // "user" or "assistant"
	Content string
	Time    time.Time
}

// Session is an in-memory chat session.
type Session struct {
	ID          string
	AgentName   string
	Messages    []Message     // display history (user + assistant text only)
	LLMHistory  []llm.Message // full LLM message history (excluding system prompt) for context replay
	CreatedAt   time.Time
	ActiveRunID string // set while an agent run is in progress
}

// runEvent is a single SSE event for an in-flight agent run.
type runEvent struct {
	typ  string // "status" | "done" | "error"
	data string
}

// agentRun tracks an in-flight agent run with fan-out to multiple SSE clients.
type agentRun struct {
	mu     sync.Mutex
	events []runEvent      // append-only replay buffer
	subs   []chan runEvent // one channel per connected SSE client
	closed bool            // true once the terminal event has been published
}

// publish appends the event to the replay buffer and fans it out to all subscribers.
func (r *agentRun) publish(evt runEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, evt)
	for _, ch := range r.subs {
		// Non-blocking: each subscriber channel is buffered; if full, drop rather than block the agent.
		select {
		case ch <- evt:
		default:
		}
	}
	if evt.typ == "done" || evt.typ == "error" {
		r.closed = true
		for _, ch := range r.subs {
			close(ch)
		}
		r.subs = nil
	}
}

// subscribe returns a channel that will receive all future events.
// Any events already in the replay buffer are sent first.
func (r *agentRun) subscribe() (chan runEvent, func()) {
	ch := make(chan runEvent, 64)
	r.mu.Lock()
	// Replay buffered events so a late-joining client catches up.
	for _, evt := range r.events {
		ch <- evt
	}
	if r.closed {
		// Run already finished — close immediately after replay.
		close(ch)
		r.mu.Unlock()
		return ch, func() {}
	}
	r.subs = append(r.subs, ch)
	r.mu.Unlock()

	unsubscribe := func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		for i, s := range r.subs {
			if s == ch {
				r.subs = append(r.subs[:i], r.subs[i+1:]...)
				break
			}
		}
	}
	return ch, unsubscribe
}

// runReporter implements agent.Reporter by publishing to an agentRun.
type runReporter struct {
	run *agentRun
}

func (r *runReporter) Update(status string) {
	r.run.publish(runEvent{typ: "status", data: status})
}

// Handler serves the web UI pages.
type Handler struct {
	store    DefinitionStore
	pool     *mcpclient.Pool
	runtime  *agent.Runtime
	tmpl     *template.Template
	mu       sync.Mutex
	sessions map[string]*Session
	runs     map[string]*agentRun
}

// NewHandler creates a new web UI handler.
func NewHandler(store DefinitionStore, runtime *agent.Runtime, pool *mcpclient.Pool) (*Handler, error) {
	funcMap := template.FuncMap{
		"renderMarkdown": renderMarkdown,
		// dict builds a map[string]any for passing multiple named values to sub-templates.
		// Keys must be strings and arguments must be provided in key-value pairs.
		// Odd-length or non-string-key arguments are silently skipped.
		"dict": func(kvs ...any) map[string]any {
			m := make(map[string]any, len(kvs)/2)
			for i := 0; i+1 < len(kvs); i += 2 {
				if key, ok := kvs[i].(string); ok {
					m[key] = kvs[i+1]
				} else {
					slog.Warn("dict: non-string key ignored in template helper", "index", i, "key", kvs[i])
				}
			}
			if len(kvs)%2 != 0 {
				slog.Warn("dict: odd number of arguments in template helper", "count", len(kvs))
			}
			return m
		},
	}
	tmpl, err := template.New("").Funcs(funcMap).ParseFS(templateFS, "templates/*.html")
	if err != nil {
		return nil, err
	}
	return &Handler{
		store:    store,
		pool:     pool,
		runtime:  runtime,
		tmpl:     tmpl,
		sessions: make(map[string]*Session),
		runs:     make(map[string]*agentRun),
	}, nil
}

// RegisterRoutes registers the web UI routes on the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /{$}", h.redirectToChat)
	mux.HandleFunc("GET /chat", h.chatPage)
	mux.HandleFunc("POST /chat/sessions", h.newSession)
	mux.HandleFunc("GET /chat/sessions/list", h.sessionListPartial)
	mux.HandleFunc("POST /chat/sessions/{id}/messages", h.postMessage)
	mux.HandleFunc("GET /chat/sessions/{id}/run", h.activeRun)
	mux.HandleFunc("GET /chat/runs/{id}/events", h.runEvents)
	mux.HandleFunc("GET /agents", h.agentsPage)
	mux.HandleFunc("GET /agents/list", h.agentListPartial)
	mux.HandleFunc("GET /agents/new", h.newAgentEditor)
	mux.HandleFunc("GET /agents/{name}/edit", h.agentEditPartial)
	mux.HandleFunc("POST /agents/yaml", h.createAgentYaml)
	mux.HandleFunc("POST /agents", h.createAgentForm)
	mux.HandleFunc("PUT /agents/{name}/yaml", h.saveAgentYaml)
	mux.HandleFunc("DELETE /agents/{name}", h.deleteAgentWeb)
	mux.HandleFunc("GET /tools", h.toolsPage)
	mux.HandleFunc("GET /tools/list", h.toolListPartial)
	mux.HandleFunc("POST /tools/generate", h.toolGeneratePartial)
	slog.Info("web UI routes registered")
}

func (h *Handler) redirectToChat(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/chat", http.StatusFound)
}

// --- Chat page ---

type chatPageData struct {
	ActivePage  string
	Agents      []*config.Definition
	Sessions    []*Session
	Current     *Session
	ActiveRunID string
}

func (h *Handler) chatPage(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("session")
	h.mu.Lock()
	sessions := h.orderedSessions()
	var current *Session
	if sessionID != "" {
		current = h.sessions[sessionID]
	}
	var activeRunID string
	if current != nil {
		activeRunID = current.ActiveRunID
	}
	h.mu.Unlock()

	data := chatPageData{
		ActivePage:  "chat",
		Agents:      h.store.ListDefinitions(),
		Sessions:    sessions,
		Current:     current,
		ActiveRunID: activeRunID,
	}

	// HTMX partial request (e.g. clicking a session in the sidebar) —
	// return only the chat content area, not the full page.
	if r.Header.Get("HX-Request") == "true" {
		h.renderPartial(w, "chat-content", data)
		return
	}

	h.render(w, "chat.html", data)
}

func (h *Handler) newSession(w http.ResponseWriter, r *http.Request) {
	agentName := r.FormValue("agent")
	if agentName == "" {
		http.Error(w, "agent is required", http.StatusBadRequest)
		return
	}

	id := newID()
	session := &Session{
		ID:        id,
		AgentName: agentName,
		Messages:  []Message{},
		CreatedAt: time.Now(),
	}

	h.mu.Lock()
	h.sessions[id] = session
	sessions := h.orderedSessions()
	h.mu.Unlock()

	if r.Header.Get("HX-Request") == "true" {
		w.Header().Set("HX-Push-Url", "/chat?session="+id)
		data := chatPageData{
			ActivePage: "chat",
			Agents:     h.store.ListDefinitions(),
			Sessions:   sessions,
			Current:    session,
		}
		// Include OOB session-list update so the sidebar refreshes without JS.
		h.renderPartial(w, "new-session-response", data)
		return
	}

	http.Redirect(w, r, "/chat?session="+id, http.StatusSeeOther)
}
func (h *Handler) postMessage(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("id")
	content := r.FormValue("message")
	if content == "" {
		http.Error(w, "message is required", http.StatusBadRequest)
		return
	}

	h.mu.Lock()
	session := h.sessions[sessionID]
	h.mu.Unlock()

	if session == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	userMsg := Message{Role: "user", Content: content, Time: time.Now()}
	h.mu.Lock()
	session.Messages = append(session.Messages, userMsg)
	h.mu.Unlock()

	// Look up agent definition
	defs := h.store.ListDefinitions()
	var def *config.Definition
	for _, d := range defs {
		if d.Name == session.AgentName {
			def = d
			break
		}
	}

	// Create a run and start the agent asynchronously.
	runID := newID()
	run := &agentRun{}

	h.mu.Lock()
	h.runs[runID] = run
	session.ActiveRunID = runID
	h.mu.Unlock()

	go func() {
		// Use a detached context so the agent run isn't cancelled when the
		// POST response is sent and the client connection closes.
		ctx := context.WithoutCancel(r.Context())
		rep := &runReporter{run: run}
		var result string
		var err error

		// Snapshot the current LLM history to pass into the agent.
		h.mu.Lock()
		history := make([]llm.Message, len(session.LLMHistory))
		copy(history, session.LLMHistory)
		h.mu.Unlock()

		var updatedHistory []llm.Message
		if def == nil {
			err = fmt.Errorf("agent %q not found", session.AgentName)
		} else {
			result, updatedHistory, err = h.runtime.RunWithReporter(ctx, def, content, rep, history)
		}

		// Append to session and publish terminal event as an HTML fragment.
		h.mu.Lock()
		if err != nil {
			slog.Error("agent run failed", "agent", session.AgentName, "error", err)
			session.Messages = append(session.Messages, Message{
				Role:    "assistant",
				Content: "Error: " + err.Error(),
				Time:    time.Now(),
			})
		} else {
			session.Messages = append(session.Messages, Message{
				Role:    "assistant",
				Content: result,
				Time:    time.Now(),
			})
			// Persist the full LLM history so subsequent messages have context.
			session.LLMHistory = updatedHistory
		}
		h.mu.Unlock()

		if err != nil {
			errHTML := fmt.Sprintf(
				`<div class="flex justify-start"><div class="msg-assistant msg-error">%s</div></div>`,
				template.HTMLEscapeString("Error: "+err.Error()),
			)
			run.publish(runEvent{typ: "done", data: errHTML})
		} else {
			doneHTML := fmt.Sprintf(
				`<div class="flex justify-start"><div class="msg-assistant">%s</div></div>`,
				string(renderMarkdown(result)),
			)
			run.publish(runEvent{typ: "done", data: doneHTML})
		}

		// Clear the active run ID on the session immediately so pollers stop finding it.
		h.mu.Lock()
		if session.ActiveRunID == runID {
			session.ActiveRunID = ""
		}
		h.mu.Unlock()

		// Keep the run in the map for a grace period so that clients which
		// auto-reconnect or connect late can still get the replay buffer.
		time.AfterFunc(30*time.Second, func() {
			h.mu.Lock()
			delete(h.runs, runID)
			h.mu.Unlock()
		})
	}()

	// Return an HTML partial with OOB user-message + thinking/SSE container.
	// The form has hx-target="#message-list" hx-swap="beforeend" so the main
	// body (thinking container) is appended after the OOB user bubble.
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	h.renderPartial(w, "post-message-response", postMessageData{
		Content:   content,
		SessionID: sessionID,
		RunID:     runID,
	})
}

type postMessageData struct {
	Content   string
	SessionID string
	RunID     string
}

// activeRun returns the current active run ID for a session, or 204 if none.
func (h *Handler) activeRun(w http.ResponseWriter, r *http.Request) {
	sessionID := r.PathValue("id")
	h.mu.Lock()
	session := h.sessions[sessionID]
	h.mu.Unlock()

	if session == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}

	h.mu.Lock()
	runID := session.ActiveRunID
	h.mu.Unlock()

	if runID == "" {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprint(w, runID)
}

// runEvents streams SSE events for a given run ID.
// Status events carry plain-text updates; the terminal "done" event carries
// an HTML fragment that replaces the thinking container client-side via
// the htmx-ext-sse extension.
func (h *Handler) runEvents(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("id")

	h.mu.Lock()
	run := h.runs[runID]
	h.mu.Unlock()

	if run == nil {
		http.Error(w, "run not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	ch, unsubscribe := run.subscribe()
	defer unsubscribe()

	for {
		select {
		case evt, open := <-ch:
			if !open {
				return
			}
			switch evt.typ {
			case "status":
				// Plain text — htmx SSE innerHTML-swaps the thinking span.
				fmt.Fprintf(w, "event: status\ndata: %s\n\n", template.HTMLEscapeString(evt.data))
			case "done":
				// evt.data is already an HTML fragment built by postMessage.
				// SSE requires each newline in the payload to be prefixed with
				// "data: " so multi-line HTML isn't truncated at the first newline.
				sseData := strings.ReplaceAll(evt.data, "\n", "\ndata: ")
				fmt.Fprintf(w, "event: done\ndata: %s\n\n", sseData)
			}
			flusher.Flush()
			if evt.typ == "done" {
				return
			}
		case <-r.Context().Done():
			return
		}
	}
}

// --- Agents page ---

type agentsPageData struct {
	ActivePage string
	Agents     []*config.Definition
}

type agentEditorData struct {
	Name    string
	RawYAML string
}

func (h *Handler) agentsPage(w http.ResponseWriter, r *http.Request) {
	data := agentsPageData{
		ActivePage: "agents",
		Agents:     h.store.ListDefinitions(),
	}
	h.render(w, "agents.html", data)
}

func (h *Handler) sessionListPartial(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	sessions := h.orderedSessions()
	h.mu.Unlock()
	data := chatPageData{
		Sessions: sessions,
	}
	h.renderPartial(w, "session-list", data)
}

func (h *Handler) agentListPartial(w http.ResponseWriter, r *http.Request) {
	data := agentsPageData{
		Agents: h.store.ListDefinitions(),
	}
	h.renderPartial(w, "agent-list-items", data)
}

func (h *Handler) agentEditPartial(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	raw, err := h.store.GetRawDefinition(name)
	if err != nil {
		http.Error(w, "agent not found", http.StatusNotFound)
		return
	}
	data := agentEditorData{
		Name:    name,
		RawYAML: string(raw),
	}
	h.renderPartial(w, "agent-editor", data)
}

// createAgentForm handles form-based agent creation from the web UI.
func (h *Handler) createAgentForm(w http.ResponseWriter, r *http.Request) {
	name := r.FormValue("name")
	systemPrompt := r.FormValue("system_prompt")
	if name == "" || systemPrompt == "" {
		http.Error(w, "name and system_prompt are required", http.StatusBadRequest)
		return
	}

	def := &config.Definition{
		Kind:         "agent",
		Name:         name,
		Description:  r.FormValue("description"),
		Model:        r.FormValue("model"),
		SystemPrompt: systemPrompt,
	}
	if err := def.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := h.store.SaveDefinition(def); err != nil {
		slog.Error("failed to save agent", "name", name, "error", err)
		http.Error(w, "failed to save", http.StatusInternalServerError)
		return
	}

	// Return updated agent list partial.
	h.renderPartial(w, "agent-list-items", agentsPageData{Agents: h.store.ListDefinitions()})
}

// newAgentEditor renders a blank YAML editor for creating a new agent.
func (h *Handler) newAgentEditor(w http.ResponseWriter, r *http.Request) {
	h.renderPartial(w, "agent-editor-new", nil)
}

// createAgentYaml handles creating a new agent from raw YAML pasted into the blank editor.
func (h *Handler) createAgentYaml(w http.ResponseWriter, r *http.Request) {
	rawYAML := r.FormValue("yaml")
	if rawYAML == "" {
		http.Error(w, "yaml is required", http.StatusBadRequest)
		return
	}
	if err := h.store.SaveRawDefinition("", []byte(rawYAML)); err != nil {
		slog.Error("failed to create agent from yaml", "error", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Parse name from the saved YAML to load the editor for the new agent.
	var def config.Definition
	if err := yamlpkg.Unmarshal([]byte(rawYAML), &def); err != nil || def.Name == "" {
		// Fall back to just refreshing the agent list.
		h.renderPartial(w, "agent-list-items", agentsPageData{Agents: h.store.ListDefinitions()})
		return
	}
	raw, _ := h.store.GetRawDefinition(def.Name)
	h.renderPartial(w, "save-yaml-response", saveYamlData{
		Editor: agentEditorData{Name: def.Name, RawYAML: string(raw)},
		Agents: h.store.ListDefinitions(),
	})
}

// saveAgentYaml handles form-based YAML save from the web UI.
func (h *Handler) saveAgentYaml(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	yaml := r.FormValue("yaml")
	if err := h.store.SaveRawDefinition(name, []byte(yaml)); err != nil {
		slog.Error("failed to save yaml", "name", name, "error", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Respond with both the updated agent list (OOB) and a fresh editor.
	raw, _ := h.store.GetRawDefinition(name)
	h.renderPartial(w, "save-yaml-response", saveYamlData{
		Editor: agentEditorData{Name: name, RawYAML: string(raw)},
		Agents: h.store.ListDefinitions(),
	})
}

type saveYamlData struct {
	Editor agentEditorData
	Agents []*config.Definition
}

// deleteAgentWeb handles web UI agent deletion.
func (h *Handler) deleteAgentWeb(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := h.store.DeleteDefinition(name); err != nil {
		slog.Error("failed to delete agent", "name", name, "error", err)
		http.Error(w, "failed to delete", http.StatusInternalServerError)
		return
	}
	// Return the updated agent list; agent row will be removed by hx-swap="outerHTML" with empty content.
	h.renderPartial(w, "agent-list-items", agentsPageData{Agents: h.store.ListDefinitions()})
}

// --- Tools page ---

// toolInfo is a view-model for a single discovered tool.
type toolInfo struct {
	QualifiedName string
	Server        string
	Name          string
	Description   string
}

type toolsPageData struct {
	ActivePage string
	Servers    []serverTools
}

type serverTools struct {
	Name  string
	Tools []toolInfo
}

func (h *Handler) toolsPage(w http.ResponseWriter, r *http.Request) {
	data := toolsPageData{
		ActivePage: "tools",
		Servers:    h.buildServerTools(),
	}
	h.render(w, "tools.html", data)
}

func (h *Handler) toolListPartial(w http.ResponseWriter, r *http.Request) {
	data := toolsPageData{
		Servers: h.buildServerTools(),
	}
	h.renderPartial(w, "tool-list", data)
}

func (h *Handler) buildServerTools() []serverTools {
	all := h.pool.ListAllTools()
	byServer := make(map[string][]toolInfo)
	for _, dt := range all {
		byServer[dt.ServerName] = append(byServer[dt.ServerName], toolInfo{
			QualifiedName: dt.QualifiedName(),
			Server:        dt.ServerName,
			Name:          dt.Tool.Name,
			Description:   dt.Tool.Description,
		})
	}
	servers := make([]serverTools, 0, len(byServer))
	for srv, tools := range byServer {
		servers = append(servers, serverTools{Name: srv, Tools: tools})
	}
	sort.Slice(servers, func(i, j int) bool { return servers[i].Name < servers[j].Name })
	return servers
}

// toolGenerateData is the view-model for the generated tool list partial.
type toolGenerateData struct {
	YAML     string
	Selected int
	Lines    int
}

func (h *Handler) toolGeneratePartial(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	tools := r.Form["tool"]
	sort.Strings(tools)

	var buf strings.Builder
	if len(tools) > 0 {
		buf.WriteString("tools:\n")
		for _, t := range tools {
			buf.WriteString("  - ")
			buf.WriteString(t)
			buf.WriteString("\n")
		}
	}

	h.renderPartial(w, "tool-generate-result", toolGenerateData{
		YAML:     buf.String(),
		Selected: len(tools),
		Lines:    len(tools) + 1, // header line + one per tool
	})
}

// --- helpers ---

func (h *Handler) render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := h.tmpl.ExecuteTemplate(w, name, data); err != nil {
		slog.Error("render template", "name", name, "error", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

func (h *Handler) renderPartial(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := h.tmpl.ExecuteTemplate(w, name, data); err != nil {
		slog.Error("render partial", "name", name, "error", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

func (h *Handler) orderedSessions() []*Session {
	out := make([]*Session, 0, len(h.sessions))
	for _, s := range h.sessions {
		out = append(out, s)
	}
	for i := 0; i < len(out)-1; i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j].CreatedAt.After(out[i].CreatedAt) {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out
}

func newID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
