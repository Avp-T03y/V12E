# V12 Engines — Presentation Script Guideline

How to write and deliver the 3-minute demo video / pitch. Keep every claim honest and consistent with the Project Playbook. The friend's competitor research drops into **Section 3**.

---

## 1. Hard rules (non-negotiable)

- **Length:** the video is **max 3 minutes** and must cover **problem → supporting data → solution → the LinuxONE demo**.
- **Honesty (this is the team's rule):**
  - It is **HIV medicine (ARVs) and test kits — never "vaccine"** (no HIV vaccine exists).
  - The demo is a **replay of real 2006–2015 shipments, shown as if live** — never call it a live 2026 feed. In deployment the model **retrains on current data**.
  - Frame impact as an **aim / proxy, not a guarantee**; predicted days-late is an **estimate**.
  - **No fabricated numbers** — every figure traces to the data audit or the model's `/metrics`.
  - **No fabricated or exaggerated competitor claims** (see Section 3).

---

## 2. The 3-minute beat sheet

| Time | Beat | Content |
|---|---|---|
| **0:00–1:00** | Opening | Hook (Uber-style ETA) → stakes (late HIV medicine = interrupted treatment) → honest problem (officers *do* sense delays, but it's manual/inconsistent and global logistics is hit by the unexpected). Use the approved 1-minute opening script. |
| **1:00–1:40** | Solution | At **order time**, two Snap ML models score each shipment (late probability + days late); a generative layer (Granite) explains **why** and recommends an action; a human makes the final call; the dashboard ranks alerts by risk. |
| **1:40–2:10** | Why IBM Z + live demo | Runs on **IBM LinuxONE (s390x)** with the **Telum NNPA accelerator**, in a secure, isolated container. Show the real evidence (lscpu / podman / machine 9175 = z17). Mark only what you verified as "Verified". |
| **2:10–2:35** | Results (honestly) | Trained on 2006–2013, evaluated on **held-out 2014–2015** (data the model never saw). Report **PR-AUC and recall**, not accuracy; show the **Type I / Type II error breakdown** (missing a real delay is the costly one). |
| **2:35–3:00** | Why us + close | The competitive edge (Section 3) → close with the opening's tagline: *"Knowing before it's late — every shipment, the moment it's ordered."* |

---

## 3. Competitive advantage — "Why us" (friend to fill in)

> **[FRIEND: add the competitor research here. Keep every claim factual and sourced — see the rules below.]**

### What to research
- **Who the alternatives are:** manual tracking (calls/emails with carriers & forwarders), generic ETA / track-and-trace tools, in-house spreadsheets, and any commercial supply-chain forecasting products.
- **Where we differ**, concretely, for each alternative.

### How to frame it honestly (rules)
- State a differentiator **only if it's verifiable**; cite the source for any external claim.
- **Don't disparage** competitors or invent their weaknesses. Compare on facts: *"Unlike X, our system does Y."*
- If a comparison isn't backed by evidence, leave it out — a weak claim a judge can poke is worse than none.

### Our honest differentiators (starting points to build on)
- **Order-time prediction**, not after-the-fact tracking — we warn when there is still runway to act.
- **Runs on IBM Z / LinuxONE** with an **on-chip AI accelerator + enterprise security**, where the sensitive data already lives — not a separate cloud GPU pipeline.
- **Explainable + human-in-the-loop** — grounded explanations and an officer who approves the action (trust, and FDA-style human oversight).
- **Rigorous, leakage-safe evaluation** — time-based holdout (train on the past, test on the future), which many quick demos skip.
- **Domain-specific signal** — e.g. the model surfaces that delays concentrate in *From RDC* shipments (17.2% vs 5.3% for Direct Drop), a health-supply-chain insight a generic tool won't capture.

---

## 4. Delivery tips

- **Show, don't tell:** spend the Why-IBM-Z beat on the *running* demo and real screenshots, not slides of text.
- **Assign speakers** per beat and rehearse the handoffs; keep a visible time check at 1:00 and 2:00.
- **Record a backup clip** of the working demo in case the live run fails during recording.
- **Lead with the summary**, then the detail — judges decide fast.
- Keep the tagline and one memorable number (e.g. "11.5% of shipments arrive late") so it sticks.

---

## 5. Do / Don't checklist

**Do**
- Say "replay of 2006–2015 data"; name real numbers from `/metrics`; show LinuxONE evidence; frame impact as an aim.

**Don't**
- Say "vaccine"; claim a live 2026 feed; guarantee impact; present predicted days-late as certain; unsourced comparisons.

---

*Submission reminders: title `[V12 Engines_IBM Z Datathon 2026 Project]` · event code SSF2026 in the project description · public repo link · 3-min video · open-source links.*
