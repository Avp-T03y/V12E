## Abstract

**V12 Engines_IBM Z Datathon 2026 Project** · Event code: **SSF2026**

V12 Engines is an **order-time delivery-delay early-warning engine** for global HIV supply chains (antiretroviral medicines and HIV test kits), running on **IBM LinuxONE**. When an order is placed, two **Snap ML** boosted-tree models predict the **probability of a late delivery** and the **estimated days late**. **IBM Granite** turns the model's top drivers into a plain-language explanation and a recommended action matched to the remaining runway (long, medium, or short). A logistics officer reviews each alert (Acknowledge, Take action, Dismiss), and that log is the audit trail.

The models are trained on **10,324 USAID/PEPFAR shipments from 2006–2015**: trained on shipments through 2013 and evaluated on held-out 2014–2015 shipments.

> [!IMPORTANT]
> This demo is a **replay of historical data, not a live 2026 feed**. In deployment, the models retrain on current data. Predicted days late is an **estimate**, and a **human always decides**.
