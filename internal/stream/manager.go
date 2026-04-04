package stream

import "sync"

const subBufferSize = 64

type Event struct {
	Type string
	Data string
}

type Stream struct {
	mu     sync.Mutex
	events []Event
	subs   []chan Event
	closed bool
}

func (s *Stream) publish(evt Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, evt)
	for _, ch := range s.subs {
		select {
		case ch <- evt:
		default:
		}
	}
	if evt.Type == "done" || evt.Type == "error" {
		s.closed = true
		for _, ch := range s.subs {
			close(ch)
		}
		s.subs = nil
	}
}

func (s *Stream) Subscribe() (<-chan Event, func()) {
	ch := make(chan Event, subBufferSize)
	s.mu.Lock()
	for _, evt := range s.events {
		ch <- evt
	}
	if s.closed {
		close(ch)
		s.mu.Unlock()
		return ch, func() {}
	}
	s.subs = append(s.subs, ch)
	s.mu.Unlock()

	unsubscribe := func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		for i, sub := range s.subs {
			if sub == ch {
				s.subs = append(s.subs[:i], s.subs[i+1:]...)
				break
			}
		}
	}
	return ch, unsubscribe
}

type Manager struct {
	mu   sync.Mutex
	runs map[string]*Stream
}

func NewManager() *Manager {
	return &Manager{runs: make(map[string]*Stream)}
}

func (m *Manager) Create(id string) *Stream {
	s := &Stream{}
	m.mu.Lock()
	m.runs[id] = s
	m.mu.Unlock()
	return s
}

func (m *Manager) Get(id string) *Stream {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.runs[id]
}

func (m *Manager) Delete(id string) {
	m.mu.Lock()
	delete(m.runs, id)
	m.mu.Unlock()
}

func (m *Manager) PublishToken(id, token string) {
	if s := m.Get(id); s != nil {
		s.publish(Event{Type: "token", Data: token})
	}
}

func (m *Manager) PublishStatus(id, status string) {
	if s := m.Get(id); s != nil {
		s.publish(Event{Type: "status", Data: status})
	}
}

func (m *Manager) PublishDone(id, html string) {
	if s := m.Get(id); s != nil {
		s.publish(Event{Type: "done", Data: html})
	}
}

func (m *Manager) PublishError(id, html string) {
	if s := m.Get(id); s != nil {
		s.publish(Event{Type: "error", Data: html})
	}
}
