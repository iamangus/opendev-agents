package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/angoo/agentfoundry/internal/api"
	"github.com/angoo/agentfoundry/internal/auth"
	"github.com/angoo/agentfoundry/internal/config"
	"github.com/angoo/agentfoundry/internal/db"
	mcpserver "github.com/angoo/agentfoundry/internal/mcp"
	"github.com/angoo/agentfoundry/internal/mcpclient"
	"github.com/angoo/agentfoundry/internal/registry"
	"github.com/angoo/agentfoundry/internal/session"
	"github.com/angoo/agentfoundry/internal/stream"
	"github.com/angoo/agentfoundry/internal/temporal"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg, err := config.LoadSystem("agentfoundry.yaml")
	if err != nil {
		slog.Warn("no agentfoundry.yaml found, using defaults", "error", err)
		cfg = config.DefaultSystem()
	}
	slog.Info("loaded system config",
		"listen", cfg.Listen,
		"definitions_dir", cfg.DefinitionsDir,
		"mcp_servers", len(cfg.MCPServers),
		"temporal_host", cfg.Temporal.HostPort,
	)

	authCfg := auth.LoadConfig()

	var (
		dbPool   *db.Pool
		jwt      *auth.JWTValidator
		groups   *auth.GroupCache
		keyStore *auth.APIKeyStore
		authMW   *auth.Middleware
	)

	if authCfg.Enabled() {
		ctx := context.Background()

		if authCfg.KeycloakAdmin.ClientID != "" && authCfg.KeycloakAdmin.ClientSecret != "" {
			groups = auth.NewGroupCache(
				authCfg.Issuer,
				authCfg.KeycloakRealm,
				authCfg.KeycloakAdmin.ClientID,
				authCfg.KeycloakAdmin.ClientSecret,
			)
			slog.Info("keycloak group cache initialized")
		} else {
			slog.Warn("keycloak admin credentials not configured, API key auth will not resolve user groups")
		}

		if os.Getenv("AUTH_DB_URL") != "" {
			dbPool, err = db.NewPool(ctx, os.Getenv("AUTH_DB_URL"))
			if err != nil {
				slog.Error("failed to connect to postgres", "error", err)
				os.Exit(1)
			}
			defer dbPool.Close()

			if err := dbPool.Migrate(ctx); err != nil {
				slog.Error("failed to run migrations", "error", err)
				os.Exit(1)
			}
		} else {
			slog.Warn("AUTH_DB_URL not set, API key management disabled")
		}

		jwt, err = auth.NewJWTValidator(ctx, authCfg)
		if err != nil {
			slog.Error("failed to initialize JWT validator", "error", err)
			os.Exit(1)
		}
		slog.Info("JWT validator initialized", "issuer", authCfg.Issuer)

		if dbPool != nil {
			keyStore = auth.NewAPIKeyStore(dbPool.Pool)
		}

		authMW = auth.NewMiddleware(jwt, keyStore, groups, authCfg)
		slog.Info("auth middleware enabled")
	} else {
		slog.Info("auth disabled (AUTH_ISSUER not set), running in open access mode")
		authMW = auth.NewMiddleware(nil, nil, nil, authCfg)
	}

	reg := registry.New()

	pool := mcpclient.NewPool()

	ctx := context.Background()
	if len(cfg.MCPServers) > 0 {
		if err := pool.Connect(ctx, cfg.MCPServers); err != nil {
			slog.Error("failed to connect to MCP servers", "error", err)
		}
	} else {
		slog.Info("no external MCP servers configured")
	}

	temporalClient, err := temporal.NewClient(cfg.Temporal.HostPort, cfg.Temporal.Namespace, cfg.Temporal.APIKey)
	if err != nil {
		slog.Error("failed to connect to temporal server", "error", err)
		os.Exit(1)
	}
	defer temporalClient.Close()

	loader := config.NewLoader(cfg.DefinitionsDir, reg)
	if err := loader.LoadAll(); err != nil {
		slog.Error("failed to load definitions", "error", err)
		os.Exit(1)
	}

	if err := loader.Watch(); err != nil {
		slog.Warn("failed to start filesystem watcher", "error", err)
	}
	defer loader.Close()

	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})

	mcpManager := mcpserver.NewManager(reg, pool, temporalClient)
	mcpManager.RegisterRoutes(mux)

	pool.OnToolsChanged(func() {
		mcpManager.RefreshAll()
	})

	streams := stream.NewManager()
	sessions := session.New()

	apiHandler := api.NewHandler(reg, pool, loader, temporalClient, streams, sessions, keyStore)
	apiHandler.RegisterRoutes(mux)

	var handler http.Handler = mux
	if authMW != nil {
		handler = authMW.Handler("/health", "/servers/")(mux)
	}

	server := &http.Server{
		Addr:    cfg.Listen,
		Handler: handler,
	}

	sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("agentfoundry daemon starting", "addr", cfg.Listen)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-sigCtx.Done()
	slog.Info("shutting down...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	mcpManager.Shutdown(shutdownCtx)
	pool.Close()
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown error", "error", err)
	}
	slog.Info("agentfoundry stopped")
}
