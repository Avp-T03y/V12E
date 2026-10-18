### **Slide 1: Title Slide**

*(On-Screen: IBM Z Datathon 2026 / V12 Engines Team)*

> "Welcome to V12 Engines. Today, we’re excited to walk you through how we’re transforming global health supply chains with an early-warning delivery engine built right on IBM LinuxONE."
> 
> 

---

### **Slide 2: 01. The Challenge**

*(On-Screen: Problem Statement & Operational Context)*

> "Think about how easy it is to track a rideshare app down to the exact minute. Yet for life-saving healthcare deliveries, logistics teams have had zero visibility the moment an order is placed.
> 
> 
> The reality is heartbreaking: when vital HIV medicines and diagnostic kits arrive late, patients face dangerous treatment gaps. Right now, officers rely on stressful, manual phone calls and reactive tracking, meaning port congestion and customs delays are found out far too late. With 11.5 percent of global health shipments arriving late—and RDC fulfillment routes spiking to 17.2 percent—our goal is to turn a 128-day median lead time into actual, proactive runway before a crisis hits."
> 
> 

---

### **Slide 3: 02. Competitive Landscape**

*(On-Screen: How order-time scoring differs from tracking and planning tools)*

> "Now, you might wonder how we compare to existing tools. Platforms like project44 and FourKites do an amazing job tracking shipments *in transit* using GPS and IoT sensors, while tools like SAP IBP focus on high-level enterprise planning.
> 
> 
> But we solve a different moment: V12 Engines predicts risk the *exact second an order is placed*, using only order attributes. That gives logistics officers the maximum possible runway to step in and act."
> 
> 

---

### **Slide 4: 03. Integrated Solution**

*(On-Screen: Order-Time Delay Warning Engine)*

> "To pull this off, our solution uses a Snap ML boosted-tree classifier running directly on IBM LinuxONE compute to rank each order's delay risk against historical ranges.
> 
> 
> We’ve built-in template explanations driven by model features, tested IBM Granite for the next evolution, and secured everything with SSH-encrypted tunnels and audit logging. An intuitive dashboard brings it all together, ranking alerts so officers can take immediate action."
> 
> 

---

### **Slide 5: 04. Technical Architecture**

*(On-Screen: End-to-End System Pipeline on IBM LinuxONE)*

> "Under the hood, our pipeline processes over 10,000 USAID SCMS shipment rows, computing rolling prior late rates. A native s390x Snap ML classifier scores risk, while FastAPI and Docker Compose power our real-time streaming replay feed for the web frontend."
> 
> 

---

### **Video Walkthrough**

*(On-Screen: Dashboard, Model, and Platform Demo)*

### **Video Walkthrough**

*(On-Screen: Dashboard, Model, and Platform Demo)*

> Here, we are looking at our live dashboard interface, replaying historical orders through our scoring API. Each incoming order is evaluated at order time, displaying real-time metrics like units at risk, median scoring latency of under 107 milliseconds, and open risk alerts." 

 [On Screen: Clicking through 'Model' tab and looking at PR-AUC & confusion matrix]

"Switching to our model tab, we evaluate performance strictly on future held-out test data. Using our Snap ML boosted-tree classifier, we achieve a strong PR-AUC compared to the no-skill baseline, optimizing for high recall to ensure we catch critical delays before they impact patients." 

 [On Screen: Clicking through 'Platform' tab showing IBM LinuxONE containers]

"Finally, looking under the hood at our Platform tab, everything runs natively on IBM LinuxONE using s390x architecture and Docker Compose containers. We feature tunnel-only SSH encryption, active scoring logs, and template-based explanations designed for transparent, secure healthcare logistics.

---

### **Slide 7: 05. Stakeholder Value**

*(On-Screen: Target Audience & Ecosystem Impact)*

> "This isn't just tech for tech's sake—it creates real impact. Logistics officers get early warnings, USAID and PEPFAR programs can protect HIV treatment continuity across 43 countries, and clinics can stop emergency rationing. Best of all, it proves that secure, powerful AI belongs right on core enterprise infrastructure."
> 
> 

---

### **Slide 8: 06. Operational Impact**

*(On-Screen: Expected Outcomes & Model Metrics)*

> "When tested on post-cutoff data with a 9.5 percent late rate, our model successfully catches 37.8 percent of late shipments at our threshold. It’s designed to help prevent supply gaps, cut costly emergency air charters by catching risks early, and bring total transparency to supply chain governance."
> 
> 

---

### **Slide 9: 07. Key Takeaways**

*(On-Screen: Learnings During the Datathon)*

> "Building this during the datathon was an incredible journey—from deploying Snap ML and testing Granite on LinuxONE without dedicated GPUs, to mastering leakage-safe temporal features under tight timelines.
> 
> 
> To close: V12 Engines—knowing before it’s late, every shipment, the moment it’s ordered."
> 
>

### **Video Walkthrough**

*(On-Screen: Dashboard, Model, and Platform Demo)*

> Here, we are looking at our live dashboard interface, replaying historical orders through our scoring API. Each incoming order is evaluated at order time, displaying real-time metrics like units at risk, median scoring latency of under 107 milliseconds, and open risk alerts." 

 [On Screen: Clicking through 'Model' tab and looking at PR-AUC & confusion matrix]

"Switching to our model tab, we evaluate performance strictly on future held-out test data. Using our Snap ML boosted-tree classifier, we achieve a strong PR-AUC compared to the no-skill baseline, optimizing for high recall to ensure we catch critical delays before they impact patients." 

 [On Screen: Clicking through 'Platform' tab showing IBM LinuxONE containers]

"Finally, looking under the hood at our Platform tab, everything runs natively on IBM LinuxONE using s390x architecture and Docker Compose containers. We feature tunnel-only SSH encryption, active scoring logs, and template-based explanations designed for transparent, secure healthcare logistics.

