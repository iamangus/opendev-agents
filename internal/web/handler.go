package web

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/angoo/agentfile/internal/config"
	"github.com/angoo/agentfile/internal/llm"
	"github.com/angoo/agentfile/internal/mcpclient"
	"github.com/angoo/agentfile/internal/stream"
	"github.com/angoo/agentfile/internal/temporal"
)

//go:embed templates/*.html
var templateFS embed.FS

type DefinitionStore interface {
	ListDefinitions() []*config.Definition
	GetRawDefinition(name string) ([]byte, error)
	SaveDefinition(def *config.Definition) error
	DeleteDefinition(name string) error
	GetDefinition(name string) *config.Definition
	SaveRawDefinition(name string, data []byte) error
}

type Message struct {
	Role    string
	Content string
	Time    time.Time
}

type Session struct {
	ID          string
	AgentName   string
	Messages    []Message
	LLMHistory  []llm.Message
	CreatedAt   time.Time
	ActiveRunID string
}

type Handler struct {
	store    DefinitionStore
	pool     *mcpclient.Pool
	temporal *temporal.Client
	streams  *stream.Manager
	tmpl     *template.Template
	mu       sync.Mutex
	sessions map[string]*Session
}

func NewHandler(store DefinitionStore, temporalClient *temporal.Client, pool *mcpclient.Pool, streams *stream.Manager) (*Handler, error) {
	funcMap := template.FuncMap{
		"renderMarkdown": renderMarkdown,
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
		"json": func(v any) template.JS {
			b, err := json.MarshalIndent(v, "", "  ")
			if err != nil {
				return template.JS("{}")
			}
			return template.JS(b)
		},
		"jsonAttr": func(v any) string {
			b, err := json.Marshal(v)
			if err != nil {
				return "[]"
			}
			return string(b)
		},
		"truncate": func(s string, max int) string {
			if len(s) <= max {
				return s
			}
			return s[:max] + "..."
		},
		"joinLines": func(ss []string) string {
			return strings.Join(ss, "\n")
		},
	}
	tmpl, err := template.New("").Funcs(funcMap).ParseFS(templateFS, "templates/*.html")
	if err != nil {
		return nil, err
	}
	return &Handler{
		store:    store,
		pool:     pool,
		temporal: temporalClient,
		streams:  streams,
		tmpl:     tmpl,
		sessions: make(map[string]*Session),
	}, nil
}

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
	mux.HandleFunc("PUT /agents/{name}", h.saveAgentForm)
	mux.HandleFunc("POST /agents/form", h.createAgentFormNew)
	mux.HandleFunc("POST /agents/{name}/clone", h.cloneAgent)
	mux.HandleFunc("DELETE /agents/{name}", h.deleteAgentWeb)
	mux.HandleFunc("GET /tools", h.toolsPage)
	mux.HandleFunc("GET /tools/list", h.toolListPartial)
	mux.HandleFunc("POST /tools/generate", h.toolGeneratePartial)
	slog.Info("web UI routes registered")
}

func (h *Handler) redirectToChat(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/chat", http.StatusFound)
}

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

	defs := h.store.ListDefinitions()
	var def *config.Definition
	for _, d := range defs {
		if d.Name == session.AgentName {
			def = d
			break
		}
	}

	runID := newID()
	h.streams.Create(runID)

	h.mu.Lock()
	session.ActiveRunID = runID
	h.mu.Unlock()

	go func() {
		ctx := context.WithoutCancel(r.Context())

		h.mu.Lock()
		history := make([]llm.Message, len(session.LLMHistory))
		copy(history, session.LLMHistory)
		h.mu.Unlock()

		var result string
		var updatedHistory []llm.Message

		if def == nil {
			h.streams.PublishError(runID, fmt.Sprintf(
				`<div class="flex justify-start"><div class="msg-assistant msg-error">%s</div></div>`,
				template.HTMLEscapeString(fmt.Sprintf("Error: agent %q not found", session.AgentName)),
			))

			h.mu.Lock()
			session.Messages = append(session.Messages, Message{
				Role:    "assistant",
				Content: fmt.Sprintf("Error: agent %q not found", session.AgentName),
				Time:    time.Now(),
			})
			if session.ActiveRunID == runID {
				session.ActiveRunID = ""
			}
			h.mu.Unlock()

			time.AfterFunc(30*time.Second, func() {
				h.streams.Delete(runID)
			})
			return
		}

		h.streams.PublishStatus(runID, "Thinking...")

		agentResult, err := h.temporal.ExecuteWorkflowSync(ctx, temporal.RunAgentParams{
			AgentName: def.Name,
			Message:   content,
			History:   history,
			StreamID:  runID,
		})
		if err != nil {
			slog.Error("agent run failed", "agent", session.AgentName, "error", err)

			h.mu.Lock()
			session.Messages = append(session.Messages, Message{
				Role:    "assistant",
				Content: "Error: " + err.Error(),
				Time:    time.Now(),
			})
			if session.ActiveRunID == runID {
				session.ActiveRunID = ""
			}
			h.mu.Unlock()

			errHTML := fmt.Sprintf(
				`<div class="flex justify-start"><div class="msg-assistant msg-error">%s</div></div>`,
				template.HTMLEscapeString("Error: "+err.Error()),
			)
			h.streams.PublishDone(runID, errHTML)

			time.AfterFunc(30*time.Second, func() {
				h.streams.Delete(runID)
			})
			return
		}

		result = agentResult.Response
		updatedHistory = agentResult.History

		h.mu.Lock()
		session.Messages = append(session.Messages, Message{
			Role:    "assistant",
			Content: result,
			Time:    time.Now(),
		})
		session.LLMHistory = updatedHistory
		if session.ActiveRunID == runID {
			session.ActiveRunID = ""
		}
		h.mu.Unlock()

		doneHTML := fmt.Sprintf(
			`<div class="flex justify-start"><div class="msg-assistant">%s</div></div>`,
			string(renderMarkdown(result)),
		)
		h.streams.PublishDone(runID, doneHTML)

		time.AfterFunc(30*time.Second, func() {
			h.streams.Delete(runID)
		})

		_ = result
	}()

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

func (h *Handler) runEvents(w http.ResponseWriter, r *http.Request) {
	runID := r.PathValue("id")

	s := h.streams.Get(runID)
	if s == nil {
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

	ch, unsubscribe := s.Subscribe()
	defer unsubscribe()

	for {
		select {
		case evt, open := <-ch:
			if !open {
				return
			}
			switch evt.Type {
			case "status":
				fmt.Fprintf(w, "event: status\ndata: %s\n\n", template.HTMLEscapeString(evt.Data))
			case "token":
				tokenData := strings.ReplaceAll(evt.Data, "\n", "\ndata: ")
				fmt.Fprintf(w, "event: token\ndata: %s\n\n", tokenData)
			case "response_start":
				fmt.Fprintf(w, "event: response_start\ndata: \n\n")
			case "done":
				sseData := strings.ReplaceAll(evt.Data, "\n", "\ndata: ")
				fmt.Fprintf(w, "event: done\ndata: %s\n\n", sseData)
			case "error":
				sseData := strings.ReplaceAll(evt.Data, "\n", "\ndata: ")
				fmt.Fprintf(w, "event: error\ndata: %s\n\n", sseData)
			}
			flusher.Flush()
			if evt.Type == "done" || evt.Type == "error" {
				return
			}
		case <-r.Context().Done():
			return
		}
	}
}

type agentsPageData struct {
	ActivePage string
	Agents     []*config.Definition
}

type agentEditorData struct {
	Def                  *config.Definition
	StructuredOutputJSON string
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
	def := h.store.GetDefinition(name)
	if def == nil {
		http.Error(w, "agent not found", http.StatusNotFound)
		return
	}
	h.renderPartial(w, "agent-editor", agentEditorData{
		Def:                  def,
		StructuredOutputJSON: structuredOutputJSON(def),
	})
}

func (h *Handler) newAgentEditor(w http.ResponseWriter, r *http.Request) {
	h.renderPartial(w, "agent-editor-new", agentEditorData{Def: &config.Definition{}})
}

func (h *Handler) saveAgentForm(w http.ResponseWriter, r *http.Request) {
	originalName := r.PathValue("name")
	def, err := definitionFromForm(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	newName := r.FormValue("name")

	if newName != originalName {
		if err := h.store.DeleteDefinition(originalName); err != nil {
			slog.Error("failed to delete old agent on rename", "old_name", originalName, "error", err)
		}
	}
	def.Name = newName

	if err := def.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := h.store.SaveDefinition(def); err != nil {
		slog.Error("failed to save agent", "name", newName, "error", err)
		http.Error(w, "failed to save", http.StatusInternalServerError)
		return
	}

	saved := h.store.GetDefinition(newName)
	if saved == nil {
		saved = def
	}
	h.renderPartial(w, "save-agent-response", saveYamlData{
		Editor: agentEditorData{Def: saved, StructuredOutputJSON: structuredOutputJSON(saved)},
		Agents: h.store.ListDefinitions(),
	})
}

func (h *Handler) createAgentFormNew(w http.ResponseWriter, r *http.Request) {
	def, err := definitionFromForm(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	def.Name = r.FormValue("name")

	if err := def.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := h.store.SaveDefinition(def); err != nil {
		slog.Error("failed to create agent", "name", def.Name, "error", err)
		http.Error(w, "failed to create", http.StatusInternalServerError)
		return
	}

	saved := h.store.GetDefinition(def.Name)
	if saved == nil {
		saved = def
	}
	h.renderPartial(w, "save-agent-response", saveYamlData{
		Editor: agentEditorData{Def: saved, StructuredOutputJSON: structuredOutputJSON(saved)},
		Agents: h.store.ListDefinitions(),
	})
}

func cloneAgentName(src string, exists func(string) bool) (string, error) {
	candidate := src + "-copy"
	if !exists(candidate) {
		return candidate, nil
	}
	for i := 2; i <= 10; i++ {
		candidate = fmt.Sprintf("%s-copy-%d", src, i)
		if !exists(candidate) {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("too many copies of %q", src)
}

func (h *Handler) cloneAgent(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	src := h.store.GetDefinition(name)
	if src == nil {
		http.Error(w, "agent not found: "+name, http.StatusNotFound)
		return
	}

	cloneName, err := cloneAgentName(name, func(s string) bool {
		return h.store.GetDefinition(s) != nil
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	clone := *src
	clone.Name = cloneName
	clone.Tools = append([]string(nil), src.Tools...)
	if src.StructuredOutput != nil {
		so := *src.StructuredOutput
		clone.StructuredOutput = &so
	}
	if err := h.store.SaveDefinition(&clone); err != nil {
		slog.Error("failed to clone agent", "source", name, "clone", cloneName, "error", err)
		http.Error(w, "failed to clone", http.StatusInternalServerError)
		return
	}

	h.renderPartial(w, "save-agent-response", saveYamlData{
		Editor: agentEditorData{Def: &clone, StructuredOutputJSON: structuredOutputJSON(&clone)},
		Agents: h.store.ListDefinitions(),
	})
}

type saveYamlData struct {
	Editor agentEditorData
	Agents []*config.Definition
}

func structuredOutputJSON(def *config.Definition) string {
	if def == nil || def.StructuredOutput == nil {
		return ""
	}
	b, err := json.MarshalIndent(def.StructuredOutput, "", "  ")
	if err != nil {
		return ""
	}
	return string(b)
}

func definitionFromForm(r *http.Request) (*config.Definition, error) {
	def := &config.Definition{
		Kind:         config.KindAgent,
		Description:  r.FormValue("description"),
		Model:        r.FormValue("model"),
		SystemPrompt: r.FormValue("system_prompt"),
		ForceJSON:    r.FormValue("force_json") != "",
	}
	if v := r.FormValue("max_turns"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			def.MaxTurns = n
		}
	}
	if v := r.FormValue("max_concurrent_tools"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			def.MaxConcurrentTools = n
		}
	}
	toolsStr := r.FormValue("tools")
	for _, line := range strings.Split(toolsStr, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			def.Tools = append(def.Tools, t)
		}
	}
	soJSON := r.FormValue("structured_output_json")
	soEnabled := r.FormValue("structured_output_enabled") == "true"
	if soEnabled && soJSON != "" {
		var so config.StructuredOutput
		if err := json.Unmarshal([]byte(soJSON), &so); err != nil {
			return nil, fmt.Errorf("invalid structured output JSON: %w", err)
		}
		def.StructuredOutput = &so
	}
	return def, nil
}

func (h *Handler) deleteAgentWeb(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := h.store.DeleteDefinition(name); err != nil {
		slog.Error("failed to delete agent", "name", name, "error", err)
		http.Error(w, "failed to delete", http.StatusInternalServerError)
		return
	}
	h.renderPartial(w, "agent-list-items", agentsPageData{Agents: h.store.ListDefinitions()})
}

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
		Lines:    len(tools) + 1,
	})
}

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
