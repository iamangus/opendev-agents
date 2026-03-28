package api

import (
	"context"
	"sync"
	"time"
)

type ToolCallStatus string

const (
	ToolCallStatusSuccess ToolCallStatus = "success"
	ToolCallStatusError   ToolCallStatus = "error"
)

// ToolCallSummary holds the AI-generated summary of a tool call.
type ToolCallSummary struct {
	Reason  string `json:"reason"`
	Outcome string `json:"outcome"`
}

type ToolCallRecord struct {
	ID        string           `json:"id"`
	Name      string           `json:"name"`
	Arguments string           `json:"arguments"`
	Result    string           `json:"result,omitempty"`
	Status    ToolCallStatus   `json:"status"`
	Error     string           `json:"error,omitempty"`
	StartedAt time.Time        `json:"started_at"`
	Duration  time.Duration    `json:"duration"`
	Summary   *ToolCallSummary `json:"summary,omitempty"`
}

type TurnRecord struct {
	TurnNumber int              `json:"turn_number"`
	Request    string           `json:"request,omitempty"`
	Response   string           `json:"response,omitempty"`
	ToolCalls  []ToolCallRecord `json:"tool_calls,omitempty"`
	StartedAt  time.Time        `json:"started_at"`
	Duration   time.Duration    `json:"duration"`
}

type RunHistory struct {
	ID             string        `json:"id"`
	Agent          string        `json:"agent"`
	Model          string        `json:"model"`
	Status         RunStatus     `json:"status"`
	UserInput      string        `json:"user_input"`
	Response       string        `json:"response,omitempty"`
	Error          string        `json:"error,omitempty"`
	AvailableTools []string      `json:"available_tools,omitempty"`
	Turns          []TurnRecord  `json:"turns,omitempty"`
	CreatedAt      time.Time     `json:"created_at"`
	CompletedAt    time.Time     `json:"completed_at,omitempty"`
	TotalDuration  time.Duration `json:"total_duration,omitempty"`

	currentTurn     *TurnRecord
	currentToolCall *ToolCallRecord
	cancel          context.CancelFunc
}

type HistoryManager struct {
	mu      sync.RWMutex
	runs    map[string]*RunHistory
	ordered []*RunHistory
	maxRuns int
}

func NewHistoryManager() *HistoryManager {
	return &HistoryManager{
		runs:    make(map[string]*RunHistory),
		ordered: make([]*RunHistory, 0),
		maxRuns: 1000,
	}
}

func (m *HistoryManager) Create(id, agent, model, userInput string, tools []string, cancel context.CancelFunc) *RunHistory {
	now := time.Now()
	h := &RunHistory{
		ID:             id,
		Agent:          agent,
		Model:          model,
		Status:         RunStatusQueued,
		UserInput:      userInput,
		AvailableTools: tools,
		Turns:          make([]TurnRecord, 0),
		CreatedAt:      now,
		cancel:         cancel,
	}

	m.mu.Lock()
	m.runs[id] = h
	m.ordered = append([]*RunHistory{h}, m.ordered...)
	if len(m.ordered) > m.maxRuns {
		removed := m.ordered[m.maxRuns:]
		for _, r := range removed {
			delete(m.runs, r.ID)
		}
		m.ordered = m.ordered[:m.maxRuns]
	}
	m.mu.Unlock()

	return h
}

func (m *HistoryManager) SetRunning(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok {
		h.Status = RunStatusRunning
	}
}

func (m *HistoryManager) StartTurn(id string, turnNum int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok {
		h.currentTurn = &TurnRecord{
			TurnNumber: turnNum,
			ToolCalls:  make([]ToolCallRecord, 0),
			StartedAt:  time.Now(),
		}
	}
}

func (m *HistoryManager) RecordRequest(id string, requestJSON string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok && h.currentTurn != nil {
		h.currentTurn.Request = requestJSON
	}
}

func (m *HistoryManager) RecordResponse(id string, responseJSON string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok && h.currentTurn != nil {
		h.currentTurn.Response = responseJSON
	}
}

func (m *HistoryManager) EndTurn(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok && h.currentTurn != nil {
		h.currentTurn.Duration = time.Since(h.currentTurn.StartedAt)
		h.Turns = append(h.Turns, *h.currentTurn)
		h.currentTurn = nil
	}
}

func (m *HistoryManager) StartToolCall(id, toolCallID, toolName, arguments string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok && h.currentTurn != nil {
		h.currentToolCall = &ToolCallRecord{
			ID:        toolCallID,
			Name:      toolName,
			Arguments: arguments,
			StartedAt: time.Now(),
		}
	}
}

func (m *HistoryManager) EndToolCall(id string, result string, status ToolCallStatus, errMsg string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok && h.currentTurn != nil && h.currentToolCall != nil {
		h.currentToolCall.Result = result
		h.currentToolCall.Status = status
		h.currentToolCall.Error = errMsg
		h.currentToolCall.Duration = time.Since(h.currentToolCall.StartedAt)
		h.currentTurn.ToolCalls = append(h.currentTurn.ToolCalls, *h.currentToolCall)
		h.currentToolCall = nil
	}
}

func (m *HistoryManager) SetCompleted(id, response string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok {
		h.Status = RunStatusCompleted
		h.Response = response
		h.CompletedAt = time.Now()
		h.TotalDuration = h.CompletedAt.Sub(h.CreatedAt)
	}
}

func (m *HistoryManager) SetFailed(id, errMsg string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if h, ok := m.runs[id]; ok {
		h.Status = RunStatusFailed
		h.Error = errMsg
		h.CompletedAt = time.Now()
		h.TotalDuration = h.CompletedAt.Sub(h.CreatedAt)
	}
}

func (m *HistoryManager) Cancel(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	h, ok := m.runs[id]
	if !ok {
		return false
	}
	switch h.Status {
	case RunStatusCompleted, RunStatusFailed, RunStatusCanceled:
		return false
	}
	if h.cancel != nil {
		h.cancel()
	}
	h.Status = RunStatusCanceled
	h.CompletedAt = time.Now()
	h.TotalDuration = h.CompletedAt.Sub(h.CreatedAt)
	return true
}

func (m *HistoryManager) Get(id string) *RunHistory {
	m.mu.RLock()
	defer m.mu.RUnlock()
	h, ok := m.runs[id]
	if !ok {
		return nil
	}
	copy := *h
	return &copy
}

func (m *HistoryManager) List(agentFilter string, statusFilter RunStatus) []*RunHistory {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*RunHistory, 0, len(m.ordered))
	for _, h := range m.ordered {
		if agentFilter != "" && h.Agent != agentFilter {
			continue
		}
		if statusFilter != "" && h.Status != statusFilter {
			continue
		}
		result = append(result, h)
	}
	return result
}

func (m *HistoryManager) ListAgents() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	seen := make(map[string]bool)
	for _, h := range m.ordered {
		seen[h.Agent] = true
	}
	agents := make([]string, 0, len(seen))
	for a := range seen {
		agents = append(agents, a)
	}
	return agents
}

// SetToolCallSummary finds the tool call record by ID (searching all turns)
// and sets its Summary field. Safe to call from a goroutine after the turn
// has been committed.
func (m *HistoryManager) SetToolCallSummary(runID, toolCallID string, s ToolCallSummary) {
	m.mu.Lock()
	defer m.mu.Unlock()
	h, ok := m.runs[runID]
	if !ok {
		return
	}
	for i := range h.Turns {
		for j := range h.Turns[i].ToolCalls {
			if h.Turns[i].ToolCalls[j].ID == toolCallID {
				h.Turns[i].ToolCalls[j].Summary = &s
				return
			}
		}
	}
}
