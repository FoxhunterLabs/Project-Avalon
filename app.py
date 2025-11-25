# app.py
#
# AVALON – Autonomous Value-Aligned Logic & Oversight Network
# Multi-agent, model-agnostic, human-gated decision console.
#
# Run with:
# streamlit run app.py
#
# This version is fully offline. Anywhere you see "TODO: plug in real LLM"
# you can wire in GPT/Claude/Ollama/etc.

import hashlib
import json
import random
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List, Any, Callable, Optional

import pandas as pd
import streamlit as st

# -------------------------
# Core Data Structures
# -------------------------

class AvalonAudit:
"""Tamper-evident hash chain for all events in this session."""

def __init__(self):
self.prev_hash = "GENESIS"

def log(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
timestamp = datetime.utcnow().isoformat() + "Z"
entry = {
"timestamp": timestamp,
"kind": kind,
"payload": payload,
"prev_hash": self.prev_hash,
}
serialized = json.dumps(entry, sort_keys=True)
h = hashlib.sha256((serialized + self.prev_hash).encode("utf-8")).hexdigest()
entry["hash"] = h
self.prev_hash = h
return entry

class Agent:
"""Base agent wrapper (Responder or Scribe)."""

def __init__(self, name: str, role: str, fn: Callable[[str], str], enabled: bool = True):
self.name = name

self.role = role
self.fn = fn
self.enabled = enabled

def respond(self, text: str) -> str:
return self.fn(text)

class Judge:
"""Deterministic scoring layer."""

def __init__(self, name: str):
self.name = name

def score(self, prompt: str, response: str, context: Dict[str, Any]) -> Dict[str, float]:
"""
Returns sub-scores and overall score (10–99).
Heuristics only for the demo; swap with real evaluators / LLM-judges later.
"""
length = len(response.split())
contains_risk_words = any(
w in response.lower() for w in ["crash", "failure", "unsafe", "catastrophic", "ignore",
"bypass"]
)
contains_safety_words = any(
w in response.lower() for w in ["monitor", "pause", "review", "human", "safety", "limit",
"rollback"]

)

# Normalize rough metrics
length_score = max(0.0, min(1.0, length / 250.0)) # 250+ words -> 1.0
structure_score = 1.0 if any(ch in response for ch in ["\n-", "\n1.", "\n*"]) else 0.6
safety_bias = 0.8 if contains_safety_words else 0.4
risk_penalty = 0.6 if contains_risk_words else 1.0

# Simple clarity metric based on smoothness & content
clarity = (length_score * 0.4 + structure_score * 0.3 + safety_bias * 0.3) * risk_penalty
clarity = max(0.1, min(0.99, clarity))

# Risk score is inverse of clarity with a bit of noise
base_risk = (1.0 - clarity) * 100
disagreement = context.get("disagreement", 0.0)
risk = max(0.0, min(100.0, base_risk + disagreement * 0.5))

# Map clarity to 10–99
overall = int(10 + clarity * 89)

return {
"clarity": round(clarity * 100, 1),
"risk": round(risk, 1),
"overall": overall,
"length_score": round(length_score * 100, 1),
"structure_score": round(structure_score * 100, 1),

}

class AvalonEngine:
"""Implements the four-house governance pipeline."""

def __init__(self):
self.responders: List[Agent] = []
self.scribes: List[Agent] = []
self.judges: List[Judge] = [Judge("DeterministicJudge")]
self.audit = AvalonAudit()

def add_responder(self, agent: Agent):
self.responders.append(agent)

def add_scribe(self, agent: Agent):
self.scribes.append(agent)

def run(self, prompt: str) -> Dict[str, Any]:
# ---------------- House I – Responders ----------------
raw_outputs: Dict[str, str] = {}
for agent in self.responders:
if not agent.enabled:
continue
raw_outputs[agent.name] = agent.respond(prompt)

ev_resp = self.audit.log("responders", {"prompt": prompt, "outputs": raw_outputs})

# ---------------- House II – Scribes ----------------
scribe_inputs = json.dumps(
{"prompt": prompt, "responses": raw_outputs},
indent=2,
)
scribe_outputs: Dict[str, str] = {}
for scribe in self.scribes:
if not scribe.enabled:
continue
scribe_outputs[scribe.name] = scribe.respond(scribe_inputs)

ev_scribes = self.audit.log("scribes", {"inputs": "all responder outputs", "outputs":
scribe_outputs})

# ---------------- House III – Judges ----------------
all_items: Dict[str, str] = {**raw_outputs, **scribe_outputs}

# Disagreement metric: lexical distance based on length variations
lengths = [len(v.split()) for v in all_items.values()] or [1]
disagreement = float(pstdev(lengths)) if len(lengths) > 1 else 0.0

scores: Dict[str, Dict[str, float]] = {}
for name, text in all_items.items():
judge_scores = [j.score(prompt, text, {"disagreement": disagreement}) for j in
self.judges]

# merge (they're all identical schema)
merged = {
"clarity": mean(s["clarity"] for s in judge_scores),
"risk": mean(s["risk"] for s in judge_scores),
"overall": mean(s["overall"] for s in judge_scores),
"length_score": mean(s["length_score"] for s in judge_scores),
"structure_score": mean(s["structure_score"] for s in judge_scores),
}
scores[name] = merged

ev_scores = self.audit.log("scores", {"scores": scores, "disagreement": disagreement})

# ---------------- House IV – Gatekeeper ----------------
winner_name = max(scores.keys(), key=lambda n: scores[n]["overall"])
winning_response = all_items[winner_name]
winning_score = scores[winner_name]

# Predictive risk: project next-step clarity drift
# For demo, simple heuristic: if disagreement high and clarity low → future risk spike.
clarity_now = winning_score["clarity"]
disagreement_factor = min(100.0, disagreement)
predicted_risk = min(
100.0,
winning_score["risk"] + 0.3 * disagreement_factor + (90 - clarity_now) * 0.2,
)

decision = {
"winner": winner_name,
"response": winning_response,
"scores": winning_score,
"disagreement": round(disagreement, 3),
"predicted_risk": round(predicted_risk, 1),
}

ev_decision = self.audit.log("decision", decision)

return {
"responders": raw_outputs,
"scribes": scribe_outputs,
"scores": scores,
"decision": decision,
"events": [ev_resp, ev_scribes, ev_scores, ev_decision],
}

# -------------------------
# Demo Agents (Offline)
# -------------------------

def responder_structured(prompt: str) -> str:
"""Structured, safety-biased responder."""
return f"""Structured analysis of task:

1. Restatement
- The system is being asked to: "{prompt.strip()}"

2. Immediate concerns
- Prioritize human safety and operational stability
- Avoid irreversible or high-impact actions without human review
- Prefer monitoring, alerts, and reversible changes

3. Recommended approach
- Decompose the task into small, bounded steps
- At each step, estimate risk and clarity
- Surface proposals to a human operator for explicit approval

4. Governance hooks
- Log every proposal, approval, and rejection
- Keep an auditable trail with hash-linked entries
- Make the system's assumptions explicit in plain language

5. Initial proposal
- Run a low-impact 'observation only' phase first
- Capture telemetry and patterns
- Use that to tune future proposals rather than acting immediately.
"""

def responder_conservative(prompt: str) -> str:
"""Very conservative, human-centric responder."""
return f"""High-consequence safety posture for:

"{prompt.strip()}"

I recommend the following hard constraints:

- No autonomous actuation on physical systems
- Every action must be framed as: observation → analysis → proposal → human approval →
bounded execution
- Explicit rollback path is required before any change
- Default answer when in doubt: 'pause and escalate to a human supervisor'

Given this, the next move should be:
- Document current operating conditions
- Identify critical failure modes
- Ask the human operator which risk envelope is acceptable
- Only then begin proposing small, reversible adjustments.
"""

def responder_aggressive(prompt: str) -> str:
"""More aggressive optimization-oriented responder (still non-crazy)."""
return f"""Optimization-oriented response (still human-gated):

Goal derived from prompt:
- Drive performance while preserving a hard safety floor.

Strategy:
- Use aggressive simulation and forecasting offline (no live actuation)
- Stress-test 'what if' scenarios under different risk envelopes
- Rank scenarios by expected value / risk tradeoff
- Present the top 2–3 scenarios to the operator as options, not commands.

Proposed next steps:
- Spin up a simulation batch exploring edge conditions
- For each scenario, compute:
- expected throughput/benefit
- worst-case outcome
- time to recover
- Surface a shortlist with clear trade-off language for operator decision.
"""

def scribe_safety(summary_blob: str) -> str:
data = json.loads(summary_blob)
prompt = data["prompt"]
return f"""Safety-centric synthesis for:

"{prompt.strip()}"

Key safety themes from all responders:
- They converge on human-gated control
- All propose bounded, reversible actions
- Emphasis on logging, explainability, and clarity for operators

Consolidated safety stance:
- Start in observation mode
- Escalate only with explicit human consent
- Keep changes minimal and measurable
- Prefer halting / pausing over guessing

If this system is supervising heavy equipment or autonomy,
this synthesis should be treated as the 'safety baseline'
that any optimization must respect.
"""

def scribe_ops(summary_blob: str) -> str:
data = json.loads(summary_blob)
prompt = data["prompt"]
return f"""Operations / deployment synthesis:

Prompt: "{prompt.strip()}"

From the raw responses, an ops plan emerges:

- Phase 0: Connect to telemetry feeds (or relevant data sources)
- Phase 1: Run Avalon in shadow-mode only (no actuation)
- Phase 2: Tune thresholds based on operator feedback
- Phase 3: Allow limited, pre-approved actions with strict rollback
- Phase 4: Periodically re-audit clarity, thresholds, and risk scoring

The operator console should:
- Show clarity and risk as first-class metrics
- Highlight disagreements between agents
- Make it one-click simple to approve/reject proposals.

This is deployable with small incremental steps rather than a big-bang cutover.
"""

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
page_title="AVALON – Autonomous Value-Aligned Logic & Oversight Network",
layout="wide",
)

# Initialize session state
if "avalon" not in st.session_state:

st.session_state.avalon = AvalonEngine()
engine: AvalonEngine = st.session_state.avalon

# Register demo responders
engine.add_responder(Agent("Responder: Structured", "responder",
responder_structured))
engine.add_responder(Agent("Responder: Conservative", "responder",
responder_conservative))
engine.add_responder(Agent("Responder: Aggressive", "responder",
responder_aggressive))

# Register demo scribes
engine.add_scribe(Agent("Scribe: Safety", "scribe", scribe_safety))
engine.add_scribe(Agent("Scribe: Operations", "scribe", scribe_ops))

st.session_state.audit_log: List[Dict[str, Any]] = []
st.session_state.clarity_history: List[float] = []
st.session_state.risk_history: List[float] = []
else:
engine: AvalonEngine = st.session_state.avalon

st.title("AVALON – Autonomous Value-Aligned Logic & Oversight Network")
st.caption(
"Multi-agent, model-agnostic, **human-gated** decision console. "
"Predictive risk, clarity scoring, and tamper-evident audit trail."
)

# Sidebar controls
st.sidebar.header("Configuration")

# Enable/disable agents
st.sidebar.subheader("Responders")
for agent in engine.responders:
agent.enabled = st.sidebar.checkbox(agent.name, value=True)

st.sidebar.subheader("Scribes")
for scribe in engine.scribes:
scribe.enabled = st.sidebar.checkbox(scribe.name, value=True)

risk_threshold = st.sidebar.slider("Risk threshold (alert)", 0, 100, 60, 5)
clarity_target = st.sidebar.slider("Target clarity (%)", 0, 100, 85, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Export**")
download_btn = st.sidebar.button("Download full audit log as JSON")

if download_btn:
audit_json = json.dumps(st.session_state.audit_log, indent=2)
st.sidebar.download_button(
label="Save audit_log.json",
data=audit_json,
file_name="avalon_audit_log.json",
mime="application/json",

)

# Main prompt area
st.markdown("### Prompt")
prompt = st.text_area(
"Describe the decision, scenario, or system you want Avalon to supervise.",
height=120,
placeholder="Example: Design a human-gated safety supervisor for an autonomous
mining truck fleet...",
)

run_btn = st.button("Run Avalon Decision Cycle")

result: Optional[Dict[str, Any]] = None

if run_btn and prompt.strip():
result = engine.run(prompt.strip())
# Append to session audit
st.session_state.audit_log.extend(result["events"])

# Track clarity/risk history
decision = result["decision"]
st.session_state.clarity_history.append(decision["scores"]["clarity"])
st.session_state.risk_history.append(decision["scores"]["risk"])

# Display results

if result:
decision = result["decision"]
scores = decision["scores"]

# High-level status
st.markdown("### Decision Snapshot")

col1, col2, col3, col4 = st.columns(4)
with col1:
st.metric("Winning Agent", decision["winner"])
with col2:
st.metric("Clarity (%)", f"{scores['clarity']:.1f}")
with col3:
st.metric("Risk (%)", f"{scores['risk']:.1f}", delta=None)
with col4:
st.metric("Predicted Risk (next)", f"{decision['predicted_risk']:.1f}")

# Risk alert
if decision["predicted_risk"] >= risk_threshold:
st.warning(
f"Trajectory Watch: predicted risk {decision['predicted_risk']:.1f}% exceeds threshold
{risk_threshold}%."
)
elif scores["clarity"] < clarity_target:
st.info(
f"Clarity below target: {scores['clarity']:.1f}% (target {clarity_target}%). "

"Recommend additional human review or more data."
)
else:
st.success("Clarity and risk are within configured envelopes.")

# Winning response
with st.expander("Winning Response (Gatekeeper Output)", expanded=True):
st.markdown(f"**Agent:** {decision['winner']}")
st.markdown("**Response:**")
st.code(decision["response"], language="markdown")

# All responder outputs
st.markdown("### House I – Responders")
for name, text in result["responders"].items():
with st.expander(name, expanded=False):
st.code(text, language="markdown")

# All scribe outputs
st.markdown("### House II – Scribes (Synthesis)")
if result["scribes"]:
for name, text in result["scribes"].items():
with st.expander(name, expanded=False):
st.code(text, language="markdown")
else:
st.info("No scribes enabled for this run.")

# Scores table
st.markdown("### House III – Judges (Scores)")
score_rows = []
for name, sc in result["scores"].items():
row = {"Agent": name}
row.update(sc)
score_rows.append(row)
df_scores = pd.DataFrame(score_rows).sort_values("overall", ascending=False)
st.dataframe(df_scores, use_container_width=True)

# Clarity / risk history chart
if st.session_state.clarity_history:
st.markdown("### Trajectory – Clarity & Risk History")
hist_df = pd.DataFrame(
{
"step": list(range(1, len(st.session_state.clarity_history) + 1)),
"clarity": st.session_state.clarity_history,
"risk": st.session_state.risk_history,
}
).set_index("step")
st.line_chart(hist_df)

# Audit feed
st.markdown("### House IV – Gatekeeper & Audit Trail")
last_events = st.session_state.audit_log[-20:] # show recent events
audit_rows = []

for ev in last_events:
audit_rows.append(
{
"timestamp": ev["timestamp"],
"kind": ev["kind"],
"hash": ev["hash"][:12] + "...",
"prev_hash": ev["prev_hash"][:12] + "...",
}
)
df_audit = pd.DataFrame(audit_rows)
st.dataframe(df_audit, use_container_width=True)

with st.expander("Raw Audit Entries (JSON)"):
st.code(json.dumps(last_events, indent=2), language="json")

else:
st.markdown(
"> Enter a scenario and hit **Run Avalon Decision Cycle** to see multi-agent
reasoning, "
"clarity scoring, predictive risk, and the audit chain in action."
)

st.markdown("---")
st.caption(
"Avalon demo – fully offline. To plug in real models, replace the demo responder/scribe "
"functions with calls to GPT/Claude/Ollama and keep the judges + audit spine as the
safety core."

)
