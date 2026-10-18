# Abstract

**V12 Engines_IBM Z Datathon 2026 Project** · Event code: **SSF2026**

## For the GitHub README (Markdown)


## Abstract

**V12 Engines_IBM Z Datathon 2026 Project** · Event code: **SSF2026**

V12 Engines is an **order-time delivery-delay early-warning engine** for global HIV supply chains (antiretroviral medicines and HIV test kits), running on **IBM LinuxONE (s390x)**. When an order is placed, a **Snap ML** boosted-tree classifier ranks its delay risk against recent orders, and the dashboard shows the runway until the promised delivery date and a historical typical delay range. Template explanations grounded in the model's top drivers say why an order was flagged; IBM Granite has been tested as a next step. Alerts are ranked by risk and order size for logistics officers, whose reviews are logged; **a human always decides**.

The model is trained on orders due 2010–2013 from **USAID/PEPFAR shipments** (10,324 shipments, 2006–2015 in total) and tested on orders due 2014–2015.

> [!IMPORTANT]
> The demo **replays historical orders as if placed today; it is not a live 2026 feed**. In deployment, the model would retrain on current data. The risk rank is not a calibrated probability, and the delay range is historical, not a per-order prediction.
