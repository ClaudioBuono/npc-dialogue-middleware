<h1 align="center">NPC Dialogue Middleware</h1>

<p align="center">
  <b>An engine-agnostic middleware that turns game state into dynamic, safe NPC dialogue using LLMs.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white" alt="Language: Python">
  <img src="https://img.shields.io/badge/Status-Complete-green" alt="Status: Complete">
  <a href="docs/NPC_Dialogue_Middleware_Documentation.pdf">
    <img src="https://img.shields.io/badge/Documentation-PDF-red?logo=adobeacrobatreader&logoColor=white" alt="Documentation: PDF">
  </a>
</p>

---

## Overview

**NPC Dialogue Middleware** acts as a bridge between a game engine and Large Language Models (LLMs). Instead of querying an LLM directly, the game engine sends game state (`GameContext`) and NPC information (`NPCContext`) over HTTP. The middleware handles prompt construction, dynamic model routing, safety filtering, and iterative validation, returning structured dialogue and player choices ready for the game UI.

> **Note:** This software is <ins>not</ins> meant to replace human creativity, but to serve as a source of inspiration for human designers.


## Documentation

Full project documentation and architecture details are available in PDF format:
- 📄 **[View Project Documentation (PDF)](docs/NPC_Dialogue_Middleware_Documentation.pdf)** *(located in [`docs/`](docs/))*

## Features

- **Engine-agnostic:** Talks over plain HTTP (locally or remotely).
- **Contextual:** Grounded in world state (`GameContext`) and character persona (`NPCContext`), supporting plain dialogue and quests.
- **Dynamic LLM routing:** Routes requests to local (e.g. Ollama) or remote (e.g. Groq/OpenAI) models based on estimated complexity.
- **Two modes:**
  - *Normal:* Full validation with an iterative **Judger + Healer** refinement loop for guaranteed JSON output.
  - *Stream:* Low-latency token streaming with static safety checks.
- **Safety and fairness:** Profanity filtering (**HurtLex** + **Aho-Corasick**), fairness guidelines, and anti-hallucination checks.
- **Guaranteed fallback:** If validation fails after maximum refinement attempts, the middleware returns an error so the engine can display pre-written fallback lines.
- **Configurable & Observable:** Hot-reloadable settings, structured logs, and telemetry snapshots (`telemetry_snapshot.json`).

## Main modules

- **Orchestrator:** Single entry point coordinating the pipeline and request lifecycle.
- **Pre-processing:** Validates input and filters forbidden terms via HurtLex.
- **Contract Builder:** Builds prompts and enforces the JSON output schema.
- **LLM Router:** Profiles models (TTFT, throughput) and routes by complexity.
- **Refiner (Judger + Healer):** The Judger evaluates quality/safety; the Healer rewrites failing dialogue.
- **Memory:** Keeps active conversation history.
- **State Manager:** Manages middleware states (`IDLE`, `STARTING`, `SETTING_CONTEXT`, `GENERATING`).
- **Telemetry & Logging:** Tracks latency, throughput, and persistent metrics.

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/Gianpyy/npc-dialogue-middleware.git
cd npc-dialogue-middleware

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example configuration files inside `src/config/`:
```bash
# Windows (PowerShell)
Copy-Item src/config/settings.example.yaml src/config/settings.yaml
Copy-Item src/config/modelconfigs.example.yaml src/config/modelconfigs.yaml

# Linux/macOS
cp src/config/settings.example.yaml src/config/settings.yaml
cp src/config/modelconfigs.example.yaml src/config/modelconfigs.yaml
```
*(Add your model endpoint or API key in `modelconfigs.yaml` if using remote models).*

---

### Option A: Run as a Python Server (HTTP)

1. **Start the server:**
   ```bash
   python src/main.py --port 8321
   ```
   Interactive Swagger docs are available at `http://127.0.0.1:8321/docs`.

2. **Set the Game Context:**
   ```bash
   curl -X POST http://127.0.0.1:8321/set-game-context \
     -H "Content-Type: application/json" \
     -d '{
       "epoch": "Medieval",
       "environment": "Crowded tavern",
       "world_state": "Ongoing civil war"
     }'
   ```

3. **Generate a Dialogue:**
   ```bash
   curl -X POST http://127.0.0.1:8321/generate-dialogue \
     -H "Content-Type: application/json" \
     -d '{
       "npc_context": {
         "name": "Thorin",
         "age": 45,
         "personality": "Gruff but warm-hearted",
         "context": "Behind the bar cleaning a mug",
         "talkativeness": "Balanced",
         "main_character_relation": "Stranger",
         "intent": {
           "type": "Dialogue",
           "must_use_expression": "Watch out for city guards tonight.",
           "number_of_options": 2
         }
       },
       "stream": false
     }'
   ```
   *(For low-latency streaming text, set `"stream": true`).*

---

### Option B: Build & Game Engine Integration

#### 1. Build the Standalone Executable
You can compile the middleware into a single executable package without requiring Python on the end-user machine:
```bash
python build.py
```
This produces an `npc_middleware/` folder containing:
- `npc_middleware.exe` (standalone binary)
- `config/` (configuration files)
- `openapi.json` (API specification for generating engine client code)

#### 2. Integrate with the Game Engine (Unity, Unreal, Godot)
1. **Sidecar Process:** Launch `npc_middleware.exe` as a child process when the game launches (or connect to a running instance).
2. **HTTP Requests:** Send standard HTTP POST requests to `http://127.0.0.1:8321/set-game-context` and `/generate-dialogue`.
3. **Safe Fallback:** If the middleware returns an error (or a network timeout occurs), catch it and display a pre-written static dialogue in the game engine.

---

## Configuration

Two YAML files in the `config/` directory:

- **`settings.yaml`:** Language, temperatures, safety/fairness filters, and maximum refinement iterations.
- **`modelconfigs.yaml`:** Declares available models (local or remote) with endpoint, tier (`low`/`high`), context size, and API keys.

---

## Authors

Developed in 2026 by [Claudio Buono](https://github.com/ClaudioBuono) and [Gianpio Silvestri](https://github.com/Gianpyy)
