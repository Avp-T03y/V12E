# V12 Engines: Video Script, Abstract & Competitor Framing

**IBM Z Datathon 2026 · Event code: SSF2026**

This document holds the final project abstract, the 3-minute demo video script, and the sourced "Why us" competitor framing. Structure follows the team's Presentation Script Guideline (5 beats, max 3:00).

> `[X]` = fill from the real `/metrics` or the Platform owner's evidence at the event. Never estimate.

**Contents:** [Abstract](#a-final-abstract-project-description) · [Beat sheet](#b-beat-sheet-target-255) · [Voiceover](#c-voiceover-script) · [Why us](#d-why-us-competitor-framing-sourced) · [Sources](#e-sources)

---

## A. Final abstract (project description)

**[V12 Engines_IBM Z Datathon 2026 Project] · Event code: SSF2026**

V12 Engines is an order-time delivery-delay early-warning engine for global HIV supply chains (antiretroviral medicines and HIV test kits), running on IBM LinuxONE. When an order is placed, two Snap ML boosted-tree models predict the probability of a late delivery and the estimated days late. IBM Granite turns the model's top drivers into a plain-language explanation and a recommended action matched to the remaining runway (long, medium, or short). A logistics officer reviews each alert (Acknowledge, Take action, Dismiss), and that log is the audit trail. The models are trained on 10,324 USAID/PEPFAR shipments from 2006–2015, trained on shipments through 2013 and evaluated on 2014–2015. The demo is a replay of historical data, not a live 2026 feed; in deployment the models retrain on current data. Predicted days late is an estimate, and a human always decides.

---

## B. Beat sheet (target 2:55)

| Time | Beat | On screen | Speaker |
|---|---|---|---|
| 0:00–1:00 | Opening: hook, stakes, honest problem, data | Split screen (ride-app countdown vs shipment map captioned "Replay of 2006–2015 data"), officer with phone/email, route chart | Vishnu |
| 1:00–1:40 | Solution | Architecture flowchart, then one alert card (probability, runway badge, drivers, explanation, action) | Vishnu |
| 1:40–2:10 | Why IBM Z + live demo | Running replay, HITL buttons, Platform view with real evidence | Platform and dashboard owners |
| 2:10–2:35 | Results | Model view: PR curve, confusion matrix, Error breakdown | Vishnu |
| 2:35–3:00 | Why us + close | Three differentiators on screen, title card with SSF2026 | Vishnu |

Show a visible time check at 1:00 and 2:00. Record a backup clip of the working demo.

---

## C. Voiceover script

### 1. OPENING (0:00–1:00)

> A ride app can tell you your car arrives in four minutes. But for life-critical health deliveries, there is still no prediction at the moment an order is placed.
>
> The stakes are treatment continuity: when HIV medicines and test kits arrive late, patients risk interrupted treatment, and interruption is linked to worse outcomes, including a higher risk of death. Our aim is to give officers earlier warning so they can act accordingly.
>
> Logistics officers do sense delays through calls and emails with carriers and forwarders. But that signal is manual and inconsistent, and global logistics is hit by the unexpected: customs holds, cancelled flights, port congestion. Warnings get missed or come too late.
>
> That is where we come in. We are V12 Engines. We used 10,324 real USAID and PEPFAR shipments from 2006 to 2015. About 11.5 percent arrived late, and the delays are not random: shipments routed From RDC make up 78 percent of all late deliveries.

### 2. SOLUTION (1:00–1:40)

> So here is the solution. The moment an order is placed, two Snap ML models score it: one predicts the probability of being late, the other estimates how many days late. IBM Granite explains why, using only the model's top drivers, and recommends an action matched to the runway: how many days remain until the promised delivery. The dashboard ranks alerts by risk, and a logistics officer makes the final call.

### 3. WHY IBM Z + LIVE DEMO (1:40–2:10)

*Caption on screen: "Replay of 2006–2015 shipments, not a live feed"*

> Here we replay historical shipments as if they were arriving live. Each alert shows probability, estimated days late, and a runway badge that turns redder as time runs out. The officer acknowledges, takes action, or dismisses, and every decision is logged. Everything runs on IBM LinuxONE: **[X: only evidence the Platform owner verified, e.g. s390x, machine type, Docker containers; mention the NNPA accelerator only if confirmed enabled]**.

### 4. RESULTS (2:10–2:35)

> We train on shipments through 2013 and test on 2014 and 2015, data the model never saw. We report PR-AUC, not accuracy: **[X]** against a no-skill baseline of 0.115, with recall of **[X]** percent at our threshold. Missing a late shipment is the costly error, so we favor recall: **[X]** percent of predictions are missed late shipments, **[X]** percent are false alarms.

### 5. WHY US + CLOSE (2:35–3:00)

> Unlike after-the-fact tracking, we warn at order time, while there is still runway to act. Our explanations are grounded and a human stays in the loop. And we evaluate on the future, not a shuffled sample. This is a replay; in deployment the models retrain on current data. V12 Engines: knowing before it's late, every shipment, the moment it's ordered.

---

## More info on "Why us":How V12 Engines Compares to Market Alternatives competitor framing (sourced)

1. Manual Tracking (Calls & Emails with Carriers / Forwarders)
What it does: Relies on human knowledge, phone calls, and email follow-ups with carriers to sense delays.
Limitations: Manual, inconsistent, and warnings arrive too late when unexpected disruptions hit (e.g., customs holds, port congestion, or cancelled flights).
>V12 Engines Advantage: Automatically scores 100% of shipments the exact moment an order is placed, eliminating manual guesswork and catching risks upfront.

2. In-Transit Tracking Platforms (project44 & FourKites)
What they do: Provide real-time visibility and predictive ETAs using active, in-transit signals—such as GPS tracking, IoT sensors, carrier telematics, and AIS ocean positioning.
Limitations: They predict during transit. Once a shipment is already en route, the window of time to reroute or prevent an inventory stockout is significantly reduced.
>V12 Engines Advantage:Predicts at order time using only order-time attributes—before the cargo ever moves.   Gives health supply chain teams maximum "runway" (lead time) to take preventative action.
Note: V12 Engines does not claim to replace or beat their in-transit ETA accuracy; rather, it complements them by addressing the crucial order-placement moment.

3. Enterprise Planning Suites (SAP Integrated Business Planning / SAP IBP)
What they do: Broad, macro-level supply chain management covering Sales & Operations Planning (S&OP), demand sensing, inventory optimization, and general network visibility.
Limitations: Designed for high-level enterprise planning rather than delivering immediate, shipment-level delay predictions.

>V12 Engines Advantage: A narrow, hyper-focused risk-alert engine tailored specifically for critical health supply chains, providing immediate actionable explanations powered by IBM Granite. 

| Alternative | What it does (per its own materials) | How we differ (safe wording) | Source |
|---|---|---|---|
| Manual tracking (calls and emails) | Human knowledge of carriers | Manual and inconsistent; ours scores every shipment at order time | Project Playbook §2 (team framing) |
| **project44** | Predictive ETAs drawing on GPS tracking, IoT sensors, carrier updates, weather feeds and historical transit patterns | Its ETAs are built on live tracking signals. Ours use only order-time attributes, so we can warn before the shipment moves | [1], [2] |
| **FourKites** | Says its Dynamic ETA for Ocean is 20% to 40% more accurate than carrier-generated ETAs; combines AIS positioning with historical and current data | Same contrast: tracking-based ETAs in transit vs our order-time score. We do not claim to beat their accuracy | [3], [4] |
| **SAP Integrated Business Planning** | Cloud planning suite: S&OP, demand forecasting and sensing, supply planning, inventory optimization | A planning tool with a broader scope; ours is a narrow, per-shipment late-risk alert for health supply chains | [5] |

**Honest framing line:** "Tracking platforms predict arrival from live shipment signals. We predict at order time, from the order's own attributes, while there is still runway to act. They solve different moments of the same problem."

**Differentiators you can say (traceable to our own work):**
- Order-time prediction with runway tiers and matching actions.
- Grounded Granite explanations with a validator and fallback, plus a human-in-the-loop log.
- Time-based holdout evaluation (train on the past, test on the future).
- A finding from our data audit: delays concentrate in From RDC (17.2% vs 5.3% for Direct Drop).
- Runs on IBM LinuxONE, using the platform's security. Say "Verified" only for what the Platform owner enabled.

**Left out on purpose:** "FDA-style oversight," "many quick demos skip evaluation," and "where sensitive data already lives."

---

## Sources

1. project44, [What is predicted ETA in supply chain](https://www.project44.com/resources/what-is-predicted-estimated-time-of-arrival-in-supply-chain/)
2. project44, [Truckload visibility](https://www.project44.com/platform/visibility/over-the-road)
3. FourKites via Business Wire, [Dynamic ETA for Ocean, Oct 2021](https://www.businesswire.com/news/home/20211019006198/en)
4. FreightWaves, [FourKites touts ocean ETAs 20%-40% more accurate](https://www.freightwaves.com/news/fourkites-touts-ocean-etas-20-40-more-accurate-than-those-from-carriers)
5. SAP, [SAP Integrated Business Planning](https://www.sap.com/products/scm/integrated-business-planning.html)
6. Interruption and outcomes (supports the opening's "linked to worse outcomes" line): [Côte d'Ivoire stockout study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2955519/) and [cART interruption and risk of AIDS or death](https://pubmed.ncbi.nlm.nih.gov/17352766/). These show an association, not a death count for late shipments. Do not quote a number from them without the source on screen.
