// Package agentlog writes structured, human-readable log files for agent runs.
// One file is created per agent under a configurable logs directory.
// All files are truncated when New() is called, so each process startup
// produces a clean set of logs for that session only.
package agentlog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const divider = "================================================================================"

// Logger manages per-agent log files.
type Logger struct {
	dir   string
	mu    sync.Mutex            // protects files map
	files map[string]*agentFile // keyed by agent name
}

// agentFile holds the open file and its write mutex.
type agentFile struct {
	mu sync.Mutex
	f  *os.File
}

// New creates the logs directory, opens (and truncates) one log file for each
// agent name provided, and returns a ready-to-use Logger.
func New(dir string, agentNames []string) (*Logger, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("agentlog: create log dir %q: %w", dir, err)
	}

	l := &Logger{
		dir:   dir,
		files: make(map[string]*agentFile, len(agentNames)),
	}

	for _, name := range agentNames {
		af, err := openFile(dir, name)
		if err != nil {
			return nil, err
		}
		l.files[name] = af
	}

	return l, nil
}

// openFile creates or truncates the log file for the given agent name.
func openFile(dir, agentName string) (*agentFile, error) {
	path := filepath.Join(dir, agentName+".log")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, fmt.Errorf("agentlog: open log file %q: %w", path, err)
	}
	return &agentFile{f: f}, nil
}

// fileFor returns the agentFile for name, creating it lazily if it wasn't
// pre-opened (e.g. an agent defined after startup).
func (l *Logger) fileFor(name string) *agentFile {
	l.mu.Lock()
	defer l.mu.Unlock()
	af, ok := l.files[name]
	if !ok {
		af, _ = openFile(l.dir, name) // best-effort; nil handled in write
		l.files[name] = af
	}
	return af
}

// RunLog is a scoped logger for a single agent run.
type RunLog struct {
	af        *agentFile
	agentName string
	model     string
	startedAt time.Time
}

// ForRun returns a RunLog scoped to one agent invocation.
func (l *Logger) ForRun(agentName, model string) *RunLog {
	return &RunLog{
		af:        l.fileFor(agentName),
		agentName: agentName,
		model:     model,
		startedAt: time.Now(),
	}
}

// write is the single low-level write path; it serialises writes per agent file.
func (rl *RunLog) write(s string) {
	if rl == nil || rl.af == nil {
		return
	}
	rl.af.mu.Lock()
	defer rl.af.mu.Unlock()
	fmt.Fprint(rl.af.f, s)
}

// writeln appends a newline after s.
func (rl *RunLog) writeln(s string) { rl.write(s + "\n") }

// ── Public logging methods ────────────────────────────────────────────────────

// Start writes the run header.
func (rl *RunLog) Start(userInput string) {
	ts := rl.startedAt.Format("2006-01-02 15:04:05")
	rl.write("\n" + divider + "\n")
	rl.writeln(fmt.Sprintf("RUN STARTED  %s", ts))
	rl.writeln(fmt.Sprintf("Agent: %s  |  Model: %s", rl.agentName, rl.model))
	rl.writeln(divider)
}

// HistorySummary logs how many prior messages are being prepended to the context.
// Only called when historyLen > 0.
func (rl *RunLog) HistorySummary(historyLen int) {
	rl.writeln(fmt.Sprintf("\n--- CONTEXT ---"))
	rl.writeln(fmt.Sprintf("  History: %d prior message(s) prepended", historyLen))
}

// Turn writes a turn header.
func (rl *RunLog) Turn(n int) {
	rl.writeln(fmt.Sprintf("\n--- TURN %d ---", n))
}

// Request logs the JSON payload sent to the LLM, excluding the tools field
// to keep log files concise.
func (rl *RunLog) Request(rawJSON []byte) {
	rl.writeln("\n[REQUEST →]")
	rl.writeln(indent(prettyJSONBytes(stripTools(rawJSON)), "  "))
}

// Response logs the full JSON payload received from the LLM.
func (rl *RunLog) Response(rawJSON []byte) {
	rl.writeln("\n[← RESPONSE]")
	rl.writeln(indent(prettyJSONBytes(rawJSON), "  "))
}

// Completed writes the run footer with timing and turn count.
func (rl *RunLog) Completed(turns int) {
	dur := time.Since(rl.startedAt).Round(time.Millisecond)
	rl.writeln(fmt.Sprintf("\n%s", divider))
	rl.writeln(fmt.Sprintf("RUN COMPLETED  %s  |  turns: %d  |  duration: %s",
		time.Now().Format("2006-01-02 15:04:05"), turns, dur))
	rl.writeln(divider)
}

// Failed writes an error footer.
func (rl *RunLog) Failed(err error) {
	dur := time.Since(rl.startedAt).Round(time.Millisecond)
	rl.writeln(fmt.Sprintf("\n%s", divider))
	rl.writeln(fmt.Sprintf("RUN FAILED  %s  |  duration: %s",
		time.Now().Format("2006-01-02 15:04:05"), dur))
	rl.writeln(fmt.Sprintf("Error: %s", err))
	rl.writeln(divider)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// stripTools removes the "tools" key from a JSON object byte slice.
// If the input is not a valid JSON object, it is returned unchanged.
func stripTools(b []byte) []byte {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil {
		return b
	}
	delete(obj, "tools")
	out, err := json.Marshal(obj)
	if err != nil {
		return b
	}
	return out
}

// indent prefixes every line of s with the given prefix string.
func indent(s, prefix string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = prefix + l
	}
	return strings.Join(lines, "\n")
}

// prettyJSONBytes re-indents a JSON byte slice; falls back to the raw string on error.
func prettyJSONBytes(b []byte) string {
	var buf bytes.Buffer
	if err := json.Indent(&buf, b, "", "  "); err != nil {
		return string(b)
	}
	return buf.String()
}
