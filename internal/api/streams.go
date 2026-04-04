package api

import (
	"encoding/json"
	"net/http"
)

type streamTokenRequest struct {
	Token string `json:"token"`
}

func (h *Handler) publishStreamToken(w http.ResponseWriter, r *http.Request) {
	streamID := r.PathValue("id")
	if streamID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "stream id is required"})
		return
	}

	var req streamTokenRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	h.streams.PublishToken(streamID, req.Token)
	w.WriteHeader(http.StatusNoContent)
}
