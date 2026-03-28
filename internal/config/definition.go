package config

// Definition is the structure parsed from a YAML file.
// It represents an agent definition.
type Definition struct {
	Kind               Kind     `yaml:"kind" json:"kind"`
	Name               string   `yaml:"name" json:"name"`
	Description        string   `yaml:"description" json:"description"`
	Model              string   `yaml:"model,omitempty" json:"model,omitempty"`
	SystemPrompt       string   `yaml:"system_prompt" json:"system_prompt"`
	Tools              []string `yaml:"tools,omitempty" json:"tools,omitempty"` // namespaced: "server.tool" or agent name
	MaxTurns           int      `yaml:"max_turns,omitempty" json:"max_turns,omitempty"`
	MaxConcurrentTools int      `yaml:"max_concurrent_tools,omitempty" json:"max_concurrent_tools,omitempty"`
	ForceJSON          bool     `yaml:"force_json,omitempty" json:"force_json,omitempty"`
}

// Kind represents the type of definition.
type Kind string

const (
	KindAgent Kind = "agent"
)

// Validate checks that the definition has required fields set.
func (d *Definition) Validate() error {
	if d.Name == "" {
		return ErrMissingName
	}
	if d.Kind == "" {
		return ErrMissingKind
	}
	if d.Kind != KindAgent {
		return ErrInvalidKind
	}
	if d.SystemPrompt == "" {
		return ErrMissingSystemPrompt
	}
	return nil
}
