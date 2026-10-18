

Slide 1: Title & Introduction
(On-Screen: Title Slide / Team Members)


Welcome to V12 Engines. Today, we are presenting our Delivery Delay Early-Warning Engine for global health supply chains, built on IBM LinuxONE.

Slide 2: 01. The Challenge
(On-Screen: Problem Statement & Operational Context)

A rideshare app can pinpoint your driver’s arrival down to the minute. Yet for life-critical healthcare deliveries, there is still zero visibility the moment an order is placed. 

The stakes here are treatment continuity. When HIV medicines and rapid diagnostic test kits arrive late, patients face life-threatening treatment stockouts. Right now, logistics officers rely on manual phone calls, emails, and reactive track-and-trace, meaning unforeseen port congestion and customs bottlenecks surface far too late. 

To prevent these treatment gaps, officers need risk scoring at order time. Looking at our data, 11.5 percent of global health shipments arrive late, with delays heavily concentrated in 'From RDC' fulfillment routes at 17.2 percent compared to 5.3 percent for Direct Drop. Traditional systems offer zero runway, leaving officers to manage with a median planned lead time of 128 days if alerted early.


Slide 3: 02. Integrated Solution
(On-Screen: Order-Time Delay Warning Engine)

To solve this, our integrated solution deploys a dual Snap ML engine on IBM LinuxONE compute to calculate late probability and estimated delayed days. 

A grounded IBM Granite generative AI layer turns model driver outputs into plain-language summaries and runway-matched actions. Everything runs with IBM Z enterprise security—including containerized execution, TLS scoring endpoints, and audit logging. Finally, our interactive dashboard ranks alerts by risk and product criticality for officer sign-off, providing actionable lead time unlike traditional post-dispatch tracking systems.


Slide 4: 03. Technical Architecture
(On-Screen: End-to-End System Pipeline on IBM LinuxONE)

Our technical pipeline is built in four key steps: First, our data integration pipeline processes 10,324 USAID SCMS shipment rows, computing 365-day rolling prior late rates per country, vendor, and route. Second, our native s390x Snap ML boosted trees evaluate the order payload using a classifier and regressor. Third, IBM Granite converts top feature drivers into structured JSON explanations matched to runway tiers. And fourth, our FastAPI scoring service streams SSE replay events to the Nginx web frontend via Podman for interactive officer review.


Slide 5: 04. Stakeholder Value
(On-Screen: Target Audience & Ecosystem Impact)

This architecture delivers massive value across the ecosystem. Logistics officers get early risk alerts and recommended actions before dispatch. USAID and PEPFAR health programs maintain HIV treatment continuity and prevent clinic stockouts. Clinics eliminate emergency stock rationing, and patients gain uninterrupted access to life-critical antiretroviral therapies and diagnostic test kits. Meanwhile, enterprise tech teams get a demonstration of secure, scalable AI inference directly on core enterprise infrastructure, while global donors benefit from audit-ready transparency.


Slide 6: 05. Operational Impact
(On-Screen: Expected Outcomes & Model Metrics)

Looking at our operational impact, we evaluate against a no-skill late baseline of 0.115, prioritizing high recall to minimize Type II errors and surface the top 10 percent highest-risk items to control alert fatigue. 

This capability prevents supply gaps across 43 countries, cuts emergency air charter costs by identifying delays at order placement, enables buffer stock pre-positioning, and ensures transparent supply chain governance with audit-ready human logs. 


Slide 7: 06. Key Takeaways
(On-Screen: Learnings During the Datathon)

Our key technical takeaways from this datathon include gaining hands-on experience deploying Snap ML boosted trees and IBM Granite on LinuxONE containers without dedicated GPUs. We mastered strict leakage-safe feature design using temporal cutoffs, implemented guardrails for IBM Granite to prevent hallucinated causes, and executed rapid cross-functional engineering under tight timelines. 

To close: Knowing before it’s late—every shipment, the moment it’s ordered.

