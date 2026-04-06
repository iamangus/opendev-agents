package auth

import "os"

type Config struct {
	Issuer        string
	Audience      string
	RolesClaim    string
	GroupsClaim   string
	TeamPrefix    string
	AdminRoles    []string
	TeamAdminRole string
	AccessRoles   []string
	KeycloakURL   string
	KeycloakRealm string
	KeycloakAdmin struct {
		ClientID     string
		ClientSecret string
	}
	InternalAPIKey string
}

func LoadConfig() *Config {
	issuer := os.Getenv("AUTH_ISSUER")
	audience := os.Getenv("AUTH_AUDIENCE")

	rolesClaim := os.Getenv("AUTH_ROLES_CLAIM")
	if rolesClaim == "" {
		rolesClaim = "realm_access.roles"
	}
	groupsClaim := os.Getenv("AUTH_GROUPS_CLAIM")
	if groupsClaim == "" {
		groupsClaim = "groups"
	}
	teamPrefix := os.Getenv("AUTH_TEAM_PREFIX")

	adminRoles := splitEnv("AUTH_ADMIN_ROLES", "opendev-admin")
	teamAdminRole := os.Getenv("AUTH_TEAM_ADMIN_ROLE")
	if teamAdminRole == "" {
		teamAdminRole = "team-admin"
	}
	accessRoles := splitEnv("AUTH_ACCESS_ROLES", "opendev-user")

	keycloakURL := os.Getenv("KEYCLOAK_URL")
	keycloakRealm := os.Getenv("KEYCLOAK_REALM")

	return &Config{
		Issuer:        issuer,
		Audience:      audience,
		RolesClaim:    rolesClaim,
		GroupsClaim:   groupsClaim,
		TeamPrefix:    teamPrefix,
		AdminRoles:    adminRoles,
		TeamAdminRole: teamAdminRole,
		AccessRoles:   accessRoles,
		KeycloakURL:   keycloakURL,
		KeycloakRealm: keycloakRealm,
		KeycloakAdmin: struct {
			ClientID     string
			ClientSecret string
		}{
			ClientID:     os.Getenv("KEYCLOAK_ADMIN_CLIENT_ID"),
			ClientSecret: os.Getenv("KEYCLOAK_ADMIN_CLIENT_SECRET"),
		},
		InternalAPIKey: os.Getenv("WORKER_API_KEY"),
	}
}

func (c *Config) Enabled() bool {
	return c.Issuer != ""
}

func splitEnv(key, defaults string) []string {
	val := os.Getenv(key)
	if val == "" {
		val = defaults
	}
	parts := make([]string, 0)
	for _, s := range splitCSV(val) {
		s = trim(s)
		if s != "" {
			parts = append(parts, s)
		}
	}
	return parts
}

func splitCSV(s string) []string {
	var parts []string
	start := 0
	inQuote := false
	for i := 0; i < len(s); i++ {
		if s[i] == '"' {
			inQuote = !inQuote
		} else if s[i] == ',' && !inQuote {
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		parts = append(parts, s[start:])
	}
	return parts
}

func trim(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}
