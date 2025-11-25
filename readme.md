````markdown
# AVALON – Autonomous Value-Aligned Logic & Oversight Network

Multi-agent, model-agnostic, **human-gated** decision console.

AVALON is a Streamlit app that runs a full “governance cycle” around any decision or scenario you describe.  
It’s fully offline by default, and designed so you can later wire in real LLMs (GPT/Claude/Ollama/etc.) behind a safety-first spine.

---

## What AVALON Does

You give it a prompt like:

> “Design a human-gated safety supervisor for an autonomous mining truck fleet.”

AVALON then runs four “houses”:

1. **House I – Responders**  
   Multiple agents generate raw responses to your prompt:
   - `Responder: Structured` – structured, safety-biased analysis  
   - `Responder: Conservative` – hard safety posture, human-centric  
   - `Responder: Aggressive` – more optimization-focused, still human-gated  

2. **House II – Scribes**  
   Scribes read **all** responder outputs and write syntheses:
   - `Scribe: Safety` – safety baseline / constraints  
   - `Scribe: Operations` – phased ops / deployment plan  

3. **House III – Judges**  
   A deterministic judge scores each response on:
   - Clarity (`clarity` %)
   - Risk (`risk` %)
   - Overall score (`overall` 10–99)
   - Length & structure scores  

   It uses simple heuristics:
   - Word count  
   - Basic structure (bullets / numbered lists)  
   - Safety vs risk language  
   - A **disagreement** metric (variation in response lengths)

4. **House IV – Gatekeeper**  
   The gatekeeper:
   - Picks the **winning** response (highest overall score)  
   - Computes a **predicted next-step risk** based on:
     - current risk
     - clarity
     - disagreement between agents

All of this is logged into a **tamper-evident hash chain** for auditability.

---

## Features

- 🔁 **Multi-agent**: Multiple responders + scribes with different personalities.
- 🧠 **Model-agnostic**: Demo agents are simple Python functions; swap them for real LLM calls.
- 🧮 **Deterministic scoring**: Heuristic judge for clarity, risk, and structure.
- 🔐 **Tamper-evident audit**: Hash-linked log (`AvalonAudit`) for every decision cycle.
- 📈 **Trajectory view**: Clarity and risk history across runs.
- 🎛️ **Human-gated**: No autonomous actuation; everything is framed as proposals for operators.

---

## Requirements

- Python **3.10+** (recommended)
- pip / virtualenv or similar

`requirements.txt`:

```txt
streamlit>=1.25.0,<2.0.0
pandas>=2.0.0,<3.0.0
````

Install:

```bash
pip install -r requirements.txt
```

---

## Running the App

From the folder containing `app.py`:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## How to Use

1. **Configure agents in the sidebar**

   * Enable / disable individual **Responders** and **Scribes** via checkboxes.
   * Set:

     * `Risk threshold (alert)` – when predicted risk should trigger a warning.
     * `Target clarity (%)` – clarity level you want decisions to hit before acting.

2. **Enter a scenario**

   * In the main text area, describe:

     * The system or environment (e.g., “autonomous haul trucks on mine site X”)
     * What AVALON is supervising / advising on.

3. **Run a decision cycle**

   * Click **“Run Avalon Decision Cycle”**.
   * AVALON will:

     * Call responders → generate raw outputs.
     * Call scribes → synthesize across outputs.
     * Score everything via the judge.
     * Select a winner and compute predicted risk.

4. **Interpret results**

   **Decision Snapshot**

   * Winning agent
   * Clarity (%)
   * Risk (%)
   * Predicted next risk
   * Status banner:

     * ⚠️ Risk above threshold → watch / escalate.
     * ℹ️ Clarity below target → get more data / human review.
     * ✅ Within envelopes → safe enough under current config.

   **House I – Responders**

   * Expand to see each raw agent response in full.

   **House II – Scribes**

   * Synthesis responses that integrate all responders.

   **House III – Judges**

   * Table of scores for each agent.

   **Trajectory – Clarity & Risk History**

   * Line chart showing how clarity/risk evolve across runs.

   **Audit Trail**

   * Recent events with timestamp, kind, and truncated hashes.
   * Full JSON of recent events under “Raw Audit Entries (JSON)”.

5. **Export audit**

   * In the sidebar, use **“Download full audit log as JSON”** to get `avalon_audit_log.json`.

---

## Architecture Overview

### Core Types

* **`AvalonAudit`**

  * Maintains a hash chain for all events in a session.
  * Each entry:

    * `timestamp`
    * `kind` (responders, scribes, scores, decision, etc.)
    * `payload`
    * `prev_hash`
    * `hash = sha256(serialized_entry + prev_hash)`

* **`Agent`**

  * Minimal wrapper for anything that takes `str -> str`:

    ```python
    Agent(name: str, role: str, fn: Callable[[str], str], enabled: bool = True)
    ```
  * Used for both **Responders** and **Scribes**.

* **`Judge`**

  * Deterministic scoring function:

    ```python
    Judge(name: str).score(prompt: str, response: str, context: Dict[str, Any]) -> Dict[str, float]
    ```
  * Returns `clarity`, `risk`, `overall`, etc.

* **`AvalonEngine`**

  * Orchestrates the four houses:

    * Responders → Scribes → Judges → Gatekeeper.
  * API:

    ```python
    result = engine.run(prompt: str) -> Dict[str, Any]
    ```
  * Returns:

    * `responders`: raw outputs
    * `scribes`: synthesized outputs
    * `scores`: per-agent scores
    * `decision`: winning agent, scores, predicted risk, disagreement
    * `events`: the audit log entries created this run

---

## Plugging In Real LLMs

This demo is **fully offline**. Anywhere you see the demo functions, you can replace them with actual model calls.

### Where to edit

Look for these demo responders:

```python
def responder_structured(prompt: str) -> str: ...
def responder_conservative(prompt: str) -> str: ...
def responder_aggressive(prompt: str) -> str: ...
```

And demo scribes:

```python
def scribe_safety(summary_blob: str) -> str: ...
def scribe_ops(summary_blob: str) -> str: ...
```

You can swap them for calls into any model:

```python
def responder_structured(prompt: str) -> str:
    # TODO: plug in real LLM
    return call_my_model(
        system_prompt="You are a structured, safety-biased analyst...",
        user_prompt=prompt,
    )
```

Just keep the **function signatures** the same:

* Responders: `fn(prompt: str) -> str`
* Scribes: `fn(summary_blob: str) -> str` where `summary_blob` is JSON containing:

  * `"prompt"`
  * `"responses"` (all responder outputs)

The rest of the safety spine (judges, gatekeeper, audit) doesn’t care which model you use, as long as it gets strings back.

---

## Safety Model / Philosophy

* **Human-gated by design**

  * The system never issues direct “commands” to physical systems.
  * Everything is framed as “observations, analyses, and proposals” for a human operator.

* **Risk-aware selection**

  * Responses with higher clarity and safety language are favored.
  * Risk words and high disagreement push risk up.

* **Tamper-evident**

  * Every decision cycle is hash-linked so you can detect log tampering.

**Important:**
This is **not** a certified safety system. Treat it as a decision-support / governance prototype.
Final responsibility and authority should always remain with a human operator.

---

## Roadmap Ideas / Extensions

* Multiple judge types (LLM judges, rule-based judges, domain-specific scoring).
* Persistence layer for long-term audit storage (DB, S3, etc.).
* Role-based access control for operators vs auditors.
* Integration with telemetry streams for “shadow mode” supervision.

---

## License

(Choose one, e.g. MIT / Apache-2.0, and drop it here.)

```markdown
MIT License – see `LICENSE` file for details.
```

---

AVALON demo – fully offline spine. Swap the demo responders/scribes for real models and keep the judges + audit chain as the safety core.

```
```
