package auth

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
)

type Middleware struct {
	jwt      *JWTValidator
	keyStore *APIKeyStore
	groups   *GroupCache
	config   *Config
}

func NewMiddleware(jwt *JWTValidator, keyStore *APIKeyStore, groups *GroupCache, config *Config) *Middleware {
	return &Middleware{
		jwt:      jwt,
		keyStore: keyStore,
		groups:   groups,
		config:   config,
	}
}

func (m *Middleware) Handler(exemptPaths ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			for _, p := range exemptPaths {
				if r.URL.Path == p {
					next.ServeHTTP(w, r)
					return
				}
			}

			if !m.config.Enabled() {
				ctx := NewContext(r.Context(), &AuthContext{
					Subject:       "anonymous",
					IsGlobalAdmin: true,
				})
				next.ServeHTTP(w, r.WithContext(ctx))
				return
			}

			ac, err := m.authenticate(r)
			if err != nil {
				slog.Warn("auth failed", "error", err)
				w.Header().Set("Content-Type", "application/json")
				if isAccessDenied(err) {
					w.WriteHeader(http.StatusForbidden)
					w.Write([]byte(`{"error":"access denied"}`))
					return
				}
				w.WriteHeader(http.StatusUnauthorized)
				w.Write([]byte(`{"error":"unauthorized"}`))
				return
			}

			ctx := NewContext(r.Context(), ac)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func (m *Middleware) authenticate(r *http.Request) (*AuthContext, error) {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return nil, ErrUnauthorized
	}

	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == authHeader {
		return nil, ErrUnauthorized
	}

	if strings.HasPrefix(token, keyPrefix) {
		return m.authenticateAPIKey(r.Context(), token)
	}
	return m.authenticateJWT(r.Context(), token)
}

func (m *Middleware) authenticateJWT(ctx context.Context, token string) (*AuthContext, error) {
	return m.jwt.Validate(ctx, token)
}

func (m *Middleware) authenticateAPIKey(ctx context.Context, fullKey string) (*AuthContext, error) {
	rec, err := m.keyStore.Validate(ctx, fullKey)
	if err != nil {
		return nil, err
	}

	var groups, roles []string
	if m.groups != nil {
		groups, err = m.groups.GetUserGroups(ctx, rec.OwnerSubject)
		if err != nil {
			slog.Error("failed to fetch user groups for api key", "owner", rec.OwnerSubject, "error", err)
			return nil, err
		}

		roles, err = m.groups.GetUserRoles(ctx, rec.OwnerSubject)
		if err != nil {
			slog.Error("failed to fetch user roles for api key", "owner", rec.OwnerSubject, "error", err)
			return nil, err
		}
	}

	isGlobalAdmin := false
	for _, adminRole := range m.config.AdminRoles {
		if containsStr(roles, adminRole) {
			isGlobalAdmin = true
			break
		}
	}

	isTeamAdmin := containsStr(roles, m.config.TeamAdminRole)

	teams := make([]string, 0, len(groups))
	for _, g := range groups {
		if g != "" && !strings.Contains(g, "/") {
			teams = append(teams, g)
		}
	}

	return &AuthContext{
		Subject:       rec.OwnerSubject,
		AuthMethod:    "api_key",
		APIKeyName:    rec.Name,
		Roles:         roles,
		Groups:        groups,
		IsGlobalAdmin: isGlobalAdmin,
		IsTeamAdmin:   isTeamAdmin,
		Teams:         teams,
	}, nil
}

var (
	ErrUnauthorized = fmtError("unauthorized")
	ErrAccessDenied = fmtError("access denied")
)

func isAccessDenied(err error) bool {
	return err == ErrAccessDenied
}
