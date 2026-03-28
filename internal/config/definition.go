package config

import (
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v3"
)

// Definition is the structure parsed from a YAML file.
// It represents an agent definition.
type Definition struct {
	Kind               Kind              `yaml:"kind" json:"kind"`
	Name               string            `yaml:"name" json:"name"`
	Description        string            `yaml:"description" json:"description"`
	Model              string            `yaml:"model,omitempty" json:"model,omitempty"`
	SystemPrompt       string            `yaml:"system_prompt" json:"system_prompt"`
	Tools              []string          `yaml:"tools,omitempty" json:"tools,omitempty"` // namespaced: "server.tool" or agent name
	MaxTurns           int               `yaml:"max_turns,omitempty" json:"max_turns,omitempty"`
	MaxConcurrentTools int               `yaml:"max_concurrent_tools,omitempty" json:"max_concurrent_tools,omitempty"`
	ForceJSON          bool              `yaml:"force_json,omitempty" json:"force_json,omitempty"`
	StructuredOutput   *StructuredOutput `yaml:"structured_output,omitempty" json:"structured_output,omitempty"`
}

// StructuredOutput configures JSON Schema constrained responses.
// It maps directly to the OpenAI json_schema response_format block.
type StructuredOutput struct {
	Name   string          `yaml:"name" json:"name"`
	Schema json.RawMessage `yaml:"schema" json:"schema"`
	Strict bool            `yaml:"strict,omitempty" json:"strict,omitempty"`
}

// UnmarshalYAML implements yaml.Unmarshaler for StructuredOutput.
// It handles converting the YAML schema node to JSON RawMessage.
func (s *StructuredOutput) UnmarshalYAML(value *yaml.Node) error {
	type plain struct {
		Name   string `yaml:"name"`
		Strict bool   `yaml:"strict"`
	}
	var p plain
	if err := value.Decode(&p); err != nil {
		return err
	}
	s.Name = p.Name
	s.Strict = p.Strict

	// Find the schema node and convert it to JSON
	for i := 0; i+1 < len(value.Content); i += 2 {
		if value.Content[i].Value == "schema" {
			schemaNode := value.Content[i+1]
			jsonBytes, err := yamlNodeToJSON(schemaNode)
			if err != nil {
				return fmt.Errorf("structured_output.schema: %w", err)
			}
			s.Schema = json.RawMessage(jsonBytes)
			break
		}
	}
	return nil
}

// MarshalYAML implements yaml.Marshaler for StructuredOutput.
// It converts the JSON schema back to a nested YAML map.
func (s StructuredOutput) MarshalYAML() (any, error) {
	var schemaMap any
	if err := json.Unmarshal(s.Schema, &schemaMap); err != nil {
		return nil, fmt.Errorf("structured_output.schema: %w", err)
	}
	return struct {
		Name   string `yaml:"name"`
		Strict bool   `yaml:"strict,omitempty"`
		Schema any    `yaml:"schema"`
	}{
		Name:   s.Name,
		Strict: s.Strict,
		Schema: schemaMap,
	}, nil
}

// yamlNodeToJSON converts a *yaml.Node to a JSON byte slice.
func yamlNodeToJSON(node *yaml.Node) ([]byte, error) {
	var v any
	if err := node.Decode(&v); err != nil {
		return nil, err
	}
	return json.Marshal(v)
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
