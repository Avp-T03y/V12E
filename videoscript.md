>Slide 1: Title Slide
(On-Screen: IBM Z Datathon 2026 / V12 Engines Team)


"Welcome to V12 Engines. Today, we are presenting our Delivery Delay Early-Warning Engine for global health supply chains, built on IBM LinuxONE." 

>Slide 2: 01. The Challenge
(On-Screen: Problem Statement & Operational Context)

"A rideshare app can pinpoint your driver’s arrival down to the minute. Yet for life-critical healthcare deliveries, there is still zero visibility the moment an order is placed. 

The stakes here are treatment continuity. When HIV medicines and rapid diagnostic test kits arrive late, patients face life-threatening treatment stockouts. Right now, logistics officers rely on manual phone calls, emails, and reactive track-and-trace, meaning unforeseen port congestion and customs bottlenecks surface far too late. 

To prevent these treatment gaps, officers need risk scoring at order time. Looking at our data, 11.5 percent of global health shipments arrive late, with delays heavily concentrated in 'From RDC' fulfillment routes at 17.2 percent compared to 5.3 percent for Direct Drop. Traditional systems offer zero runway, leaving officers to manage with a median planned lead time of 128 days if alerted early." 

>Slide 3: 02. Competitive Landscape
(On-Screen: How order-time scoring differs from tracking and planning tools)


"Looking at the competitive landscape, tracking tools like project44 and FourKites build predictive ETAs from live tracking signals such as GPS, IoT sensors, carrier updates, and AIS during transit. Meanwhile, enterprise suites like SAP IBP focus on broad macro-level cloud planning. 

Unlike these after-the-fact or broad planning tools, V12 Engines provides a narrow, per-order late-risk alert for health supply chains using only order-time attributes—giving officers maximum runway while there is still time to act." 

>Slide 4: 03. Integrated Solution
(On-Screen: Order-Time Delay Warning Engine)


"To achieve this, our integrated solution uses a Snap ML boosted-tree classifier on IBM LinuxONE compute to rank each order's delay risk against historical delay ranges. 

We use template explanations grounded in the model’s drivers, with IBM Granite integrated and tested as the next step. Everything runs with IBM Z enterprise security—offering tunnel-only access encrypted over SSH, alongside officer review logging. Finally, our interactive dashboard ranks alerts by risk and order size for officer sign-off." 

>Slide 5: 04. Technical Architecture
(On-Screen: End-to-End System Pipeline on IBM LinuxONE)


"Our technical pipeline operates in four stages: First, our data integration pipeline processes 10,324 USAID SCMS shipment rows and computes 365-day rolling prior late rates. Second, a native s390x Snap ML boosted-tree classifier ranks order delay risk. Third, structured template explanations are generated from feature drivers, with IBM Granite tested for future deployment. And fourth, our FastAPI scoring service streams real-time replay events to the Nginx web frontend via Docker Compose across three containers for interactive officer review." 

>Video start

"Here, we are looking at our live dashboard interface, replaying historical orders through our scoring API. Each incoming order is evaluated at order time, displaying real-time metrics like units at risk, median scoring latency of under 107 milliseconds, and open risk alerts." 
 [On Screen: Clicking through 'Model' tab and looking at PR-AUC & confusion matrix]
"Switching to our model tab, we evaluate performance strictly on future held-out test data. Using our Snap ML boosted-tree classifier, we achieve a strong PR-AUC compared to the no-skill baseline, optimizing for high recall to ensure we catch critical delays before they impact patients." 
 [On Screen: Clicking through 'Platform' tab showing IBM LinuxONE containers]
"Finally, looking under the hood at our Platform tab, everything runs natively on IBM LinuxONE using s390x architecture and Docker Compose containers. We feature tunnel-only SSH encryption, active scoring logs, and template-based explanations designed for transparent, secure healthcare logistics." 

>Slide 7: 05. Stakeholder Value
(On-Screen: Target Audience & Ecosystem Impact)


"This architecture is designed to deliver massive value across the ecosystem. Logistics officers get early risk alerts and recommended actions before dispatch. USAID and PEPFAR health programs can maintain HIV treatment continuity and avoid clinic stockouts. Clinics can reduce emergency stock rationing, helping patients gain uninterrupted access to life-critical antiretroviral therapies and diagnostic test kits. Meanwhile, enterprise tech teams get a demonstration of secure, scalable AI inference directly on core enterprise infrastructure, while global donors benefit from transparent officer review logs." 

>Slide 8: 06. Operational Impact
(On-Screen: Expected Outcomes & Model Metrics)


"Evaluating on orders placed after our training cutoff where the late rate is 9.5 percent, our model achieves a PR-AUC of 0.28 against a no-skill baseline of 0.09. At our scoring threshold, we catch 37.8 percent of late shipments, with 29.1 percent of generated alerts resulting in late deliveries. 

This platform is designed to help prevent supply gaps across 43 countries, could reduce emergency air charter costs by identifying risks at order placement, enables buffer stock pre-positioning, and ensures transparent supply chain governance with logged human decisions." 

>Slide 9: 07. Key Takeaways
(On-Screen: Learnings During the Datathon)


"Our key technical takeaways from this datathon include gaining hands-on experience deploying Snap ML boosted trees and testing IBM Granite on LinuxONE containers without dedicated GPUs. We mastered strict leakage-safe feature design using temporal cutoffs, implemented grounded explanations for feature drivers, and executed rapid cross-functional engineering under tight timelines. 

To close: Knowing before it’s late—every shipment, the moment it’s ordered." 

