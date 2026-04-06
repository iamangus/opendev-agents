package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/coreos/go-oidc"
)

type JWTValidator struct {
	provider *oidc.Provider
	verifier *oidc.IDTokenVerifier
	config   *Config

	mu     sync.RWMutex
	keySet *oidc.KeySet
}

func NewJWTValidator(ctx context.Context, config *Config) (*JWTValidator, error) {
	if !config.Enabled() {
		return nil, nil
	}

	provider, err := oidc.NewProvider(ctx, config.Issuer)
	if err != nil {
		return nil, err
	}

	verifier := provider.Verifier(&oidc.Config{
		ClientID:          config.Audience,
		SkipClientIDCheck: config.Audience == "",
	})

	v := &JWTValidator{
		provider: provider,
		verifier: verifier,
		config:   config,
	}

	return v, nil
}

func (v *JWTValidator) Validate(ctx context.Context, tokenString string) (*AuthContext, error) {
	token, err := v.verifier.Verify(ctx, tokenString)
	if err != nil {
		return nil, err
	}

	var claims map[string]any
	if err := token.Claims(&claims); err != nil {
		return nil, err
	}

	return v.buildAuthContext(claims)
}

func (v *JWTValidator) buildAuthContext(claims map[string]any) (*AuthContext, error) {
	sub, _ := claims["sub"].(string)
	preferredUsername, _ := claims["preferred_username"].(string)
	email, _ := claims["email"].(string)
	azp, _ := claims["azp"].(string)

	roles := ExtractNestedClaim(claims, v.config.RolesClaim)
	groups := ExtractNestedClaim(claims, v.config.GroupsClaim)

	isGlobalAdmin := false
	for _, adminRole := range v.config.AdminRoles {
		if containsStr(roles, adminRole) {
			isGlobalAdmin = true
			break
		}
	}

	isTeamAdmin := containsStr(roles, v.config.TeamAdminRole)

	teams := make([]string, 0, len(groups))
	for _, g := range groups {
		team := strings.TrimPrefix(g, "/")
		if team != "" && !strings.Contains(team, "/") {
			if v.config.TeamPrefix != "" {
				if !strings.HasPrefix(team, v.config.TeamPrefix) {
					continue
				}
				team = strings.TrimPrefix(team, v.config.TeamPrefix)
			}
			if team != "" {
				teams = append(teams, team)
			}
		}
	}

	hasAccess := false
	for _, accessRole := range v.config.AccessRoles {
		if containsStr(roles, accessRole) {
			hasAccess = true
			break
		}
	}

	if !hasAccess {
		return nil, ErrAccessDenied
	}

	authMethod := "oidc"
	if azp != "" && sub == "" {
		authMethod = "client_credentials"
		sub = azp
	}

	return &AuthContext{
		Subject:       sub,
		Username:      preferredUsername,
		Email:         email,
		AuthMethod:    authMethod,
		Roles:         roles,
		Groups:        groups,
		IsGlobalAdmin: isGlobalAdmin,
		IsTeamAdmin:   isTeamAdmin,
		Teams:         teams,
	}, nil
}

func ExtractNestedClaim(claims map[string]any, path string) []string {
	parts := strings.Split(path, ".")
	var current any = claims
	for _, part := range parts {
		m, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current = m[part]
		if current == nil {
			return nil
		}
	}

	switch v := current.(type) {
	case []any:
		result := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				result = append(result, s)
			}
		}
		return result
	case string:
		return []string{v}
	default:
		return nil
	}
}

type tokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

func (v *JWTValidator) ExchangeForAdminToken(ctx context.Context) (string, error) {
	tokenURL := v.config.Issuer + "/protocol/openid-connect/token"

	data := "grant_type=client_credentials" +
		"&client_id=" + v.config.KeycloakAdmin.ClientID +
		"&client_secret=" + v.config.KeycloakAdmin.ClientSecret

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmtError("keycloak token exchange: %s", resp.Status)
	}

	var tr tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return "", err
	}

	return tr.AccessToken, nil
}

func containsStr(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func fmtError(format string, args ...any) error {
	slog.Error("auth error", "msg", format, "args", args)
	return fmt.Errorf(format, args...)
}
