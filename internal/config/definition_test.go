package config_test

import (
	"encoding/json"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/angoo/agentfile/internal/config"
)

func TestStructuredOutput_YAMLRoundTrip(t *testing.T) {
	input := `kind: agent
name: test-agent
system_prompt: You are a test agent.
structured_output:
    name: result
    strict: true
    schema:
        type: object
        properties:
            score:
                type: integer
        required:
            - score
        additionalProperties: false
`
	var def config.Definition
	if err := yaml.Unmarshal([]byte(input), &def); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if def.StructuredOutput == nil {
		t.Fatal("expected StructuredOutput to be set")
	}
	if def.StructuredOutput.Name != "result" {
		t.Errorf("got Name=%q, want %q", def.StructuredOutput.Name, "result")
	}
	if !def.StructuredOutput.Strict {
		t.Error("expected Strict=true")
	}
	// Schema should round-trip as valid JSON
	var schema map[string]any
	if err := json.Unmarshal(def.StructuredOutput.Schema, &schema); err != nil {
		t.Errorf("Schema is not valid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("got schema.type=%v, want object", schema["type"])
	}
}

func TestDefinition_StructuredOutputNilByDefault(t *testing.T) {
	input := `kind: agent
name: simple
system_prompt: Hello.
`
	var def config.Definition
	if err := yaml.Unmarshal([]byte(input), &def); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if def.StructuredOutput != nil {
		t.Errorf("expected StructuredOutput to be nil, got %+v", def.StructuredOutput)
	}
}
