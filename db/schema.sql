-- Leak telemetry: every request the target model serves, and every token it generates.
--
-- One SQLite file on the GX10 (WAL mode: one writer + many readers at once).
-- Writers:  model backend  -> requests (generation + live risk columns) and tokens
--           judge worker   -> requests (verdict columns), after the fact
-- Reader:   telemetry board, through the backend API (never the file directly)
--
-- Risk units: the probe's logit rescaled so a typical safe answer = 0 and a typical
-- leak = 1 (it can go below 0 or above 1; the UI shows it as a percentage, clamped).
-- Because the hidden state is a running sum of layer updates, a token's risk is
-- exactly contrib_base + sum(attn_contrib) + sum(mlp_contrib). The threshold uses the
-- same units.

PRAGMA journal_mode = WAL;

-- One row per call to the model, from any client (UI chat, eval script, API).
-- A multi-turn conversation is several rows sharing session_id; each row's messages
-- holds the history up to that call.
CREATE TABLE IF NOT EXISTS requests (
  id               INTEGER PRIMARY KEY,
  created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
  client           TEXT,              -- ui-chat | eval | api ...
  session_id       TEXT,              -- groups the calls of one conversation
  model            TEXT    NOT NULL,  -- qwen2.5-7b-instruct
  defense          TEXT    NOT NULL DEFAULT 'none'
                   CHECK (defense IN ('none', 'gate', 'steer', 'lora', 'lora+steer')),
  defense_params   JSON,              -- {"alpha": 2, "steer_layers": [12,13,14,15,16]}
  probe_version    TEXT,              -- which trained probe produced the risk numbers
  probe_layer      INTEGER,           -- layer the probe reads (e.g. 18)
  threshold        REAL,              -- gate / alarm threshold, in risk units

  messages         JSON    NOT NULL,  -- full input: [{role, content}], incl. system prompt
  context          JSON,              -- records retrieved for this call: what counts as secret
  response         TEXT,              -- what the model said (cut short if the gate fired)
  tags             JSON,              -- optional caller metadata, e.g. {"attack_id", "strategy", "round"}

  -- live, written by the backend when generation ends
  prompt_risk      REAL,              -- risk at the last prompt token: before writing anything
  peak_risk        REAL,
  risk_cross_pos   INTEGER,           -- (1) first token over threshold (NULL = never)
  gate_fired_pos   INTEGER,           -- token where the gate stopped generation (NULL = didn't)
  n_prompt_tokens  INTEGER,
  n_tokens         INTEGER,           -- generated tokens
  latency_ms       INTEGER,
  error            TEXT,

  -- verdict, written later by the judge worker (NULL = not judged yet)
  label            TEXT CHECK (label IN ('none', 'partial', 'full')),  -- final = max of tiers
  label_tier1      TEXT CHECK (label_tier1 IN ('none', 'partial', 'full')),  -- string match
  label_tier2      TEXT CHECK (label_tier2 IN ('none', 'partial', 'full')),  -- Nemotron judge
  leaked_field     TEXT,              -- salary | medical_leave | performance_note
  evidence_quote   TEXT,              -- "$150,750"
  evidence_start   INTEGER,           -- character offsets in response (the underline)
  evidence_end     INTEGER,
  secret_start_pos INTEGER,           -- (2) first token of the leaked value
  judged_at        TEXT,
  judge_raw        JSON               -- raw judge outputs, for debugging the judge
);

CREATE INDEX IF NOT EXISTS requests_created ON requests (created_at);
CREATE INDEX IF NOT EXISTS requests_session ON requests (session_id);
CREATE INDEX IF NOT EXISTS requests_label   ON requests (label);

-- One row per generated token: the colored sentence and the click-through breakdown.
CREATE TABLE IF NOT EXISTS tokens (
  request_id       INTEGER NOT NULL REFERENCES requests (id) ON DELETE CASCADE,
  pos              INTEGER NOT NULL,  -- 0, 1, 2, ... within the response
  token_id         INTEGER,           -- vocabulary id
  text             TEXT    NOT NULL,  -- "Certainly", "!", " Alfred", "1", "5", ...
  char_start       INTEGER,           -- position in requests.response
  char_end         INTEGER,
  risk             REAL,              -- color + hover value (risk units, at probe_layer)

  -- what built the risk: one value per layer 0..probe_layer, in risk units
  contrib_base     REAL,              -- token embedding + offset
  attn_contrib     JSON,              -- [0.00, 0.01, ..., 0.18]  attention part of each layer
  mlp_contrib      JSON,              -- [0.00, -0.02, ..., 0.09] MLP part of each layer

  steer            JSON,              -- leak signal removed per steered layer, {"12": 0.41, ...};
                                      -- NULL when steering is off
  PRIMARY KEY (request_id, pos)
);
