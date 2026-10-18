Slide 1: Title Slide
(On-Screen: IBM Z Datathon 2026 / V12 Engines Team)

"Welcome to V12 Engines. We built an order-time delivery-delay early-warning engine for global health supply chains, running on IBM LinuxONE."

>Slide 2: 01. The Challenge
(On-Screen: Problem Statement & Operational Context)

"A rideshare app can tell you when your driver will arrive, almost to the minute. Health supply chains rarely get that warning when an order is placed. In 10,324 USAID and PEPFAR shipments, 11.5 percent arrived late, and 17.2 percent of From RDC orders. For HIV medicines and test kits, that can interrupt treatment. Yet the median planned lead time is 128 days. Warn at order time, and that becomes runway to act."

>Slide 3: 02. Competitive Landscape
(On-Screen: How order-time scoring differs from tracking and planning tools)
"Tools like project44 and FourKites predict arrival while a shipment is in transit. SAP IBP covers broad planning. We focus on an earlier moment: the order itself, using only what is known when it is placed."

>Slide 4: 03. Integrated Solution
(On-Screen: Order-Time Delay Warning Engine)
"A Snap ML boosted-tree model ranks each order against the previous 30 days: the top 10 percent are high risk, the next 15 percent elevated. Each alert explains its top drivers and suggests an action matched to the runway. IBM Granite is installed and tested, but off in the demo for speed. It all runs in containers behind an SSH tunnel, and a human always decides."

>Slide 5: 04. Technical Architecture
(On-Screen: End-to-End System Pipeline on IBM LinuxONE)
"Features use only deliveries completed before each order: how late its vendor, country and route have been over the past year. FastAPI scores each order and streams it to the dashboard."

>Slide 6: Video Walkthrough
(On-Screen: Live tab)
"Here is our live dashboard, replaying 2014 to 2015 orders through the scoring API as if placed today, with units at risk, open alerts, and a median scoring latency of about 107 milliseconds."
[On Screen: Model tab, PR-AUC and confusion matrix]
"On the Model tab, we evaluate on held-out test data. PR-AUC is well above the no-skill baseline, and the confusion matrix shows what we catch and miss within a fixed alert budget."
[On Screen: Platform tab, IBM LinuxONE containers]
"The Platform tab shows it all running natively on IBM LinuxONE, s390x, in Docker Compose containers, with measured latency and template explanations."

>Slide 7: 05. Stakeholder Value
(On-Screen: Target Audience & Ecosystem Impact)
"Officers get a short, ranked list of risky orders. Programs like USAID and PEPFAR can act before delays cause stockouts, and every review is logged."

>Slide 8: 06. Operational Impact
(On-Screen: Expected Outcomes & Model Metrics)
"On orders placed after the training data, the model catches 37.8 percent of late shipments while flagging 12.3 percent of orders, and its alerts are late three times as often as average. That is comparable to the best simple rule, and real-world impact is still to be measured."

>Slide 9: 07. Key Takeaways
(On-Screen: Learnings During the Datathon)
"We learned to run Snap ML and Granite on s390x without a GPU, and to use only what is known at order time. Finding 811 test orders that overlapped training, we re-evaluated on clean data. V12 Engines: knowing before it's late, every shipment, the moment it's ordered."
 
 
