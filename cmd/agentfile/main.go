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

	"github.com/angoo/agentfile/internal/api"
	"github.com/angoo/agentfile/internal/config"
	mcpserver "github.com/angoo/agentfile/internal/mcp"
	"github.com/angoo/agentfile/internal/mcpclient"
	"github.com/angoo/agentfile/internal/registry"
	"github.com/angoo/agentfile/internal/stream"
	"github.com/angoo/agentfile/internal/temporal"
	"github.com/angoo/agentfile/internal/web"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	cfg, err := config.LoadSystem("agentfile.yaml")
	if err != nil {
		slog.Warn("no agentfile.yaml found, using defaults", "error", err)
		cfg = config.DefaultSystem()
	}
	slog.Info("loaded system config",
		"listen", cfg.Listen,
		"definitions_dir", cfg.DefinitionsDir,
		"mcp_servers", len(cfg.MCPServers),
		"temporal_host", cfg.Temporal.HostPort,
	)

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

	apiHandler := api.NewHandler(reg, pool, loader, temporalClient, streams)
	apiHandler.RegisterRoutes(mux)

	webHandler, err := web.NewHandler(loader, temporalClient, pool, streams)
	if err != nil {
		slog.Error("failed to create web UI handler", "error", err)
		os.Exit(1)
	}
	webHandler.RegisterRoutes(mux)

	server := &http.Server{
		Addr:    cfg.Listen,
		Handler: mux,
	}

	sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("agentfile daemon starting", "addr", cfg.Listen)
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
	slog.Info("agentfile stopped")
}
