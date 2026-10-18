Here is the shortened, punchy version of your script, trimmed down to fit smoothly within a 3-minute video timeline while retaining all key technical metrics and data points:

---

### **Slide 1: Title Slide**

*(On-Screen: IBM Z Datathon 2026 / V12 Engines Team)*

> "Welcome to V12 Engines. We are presenting our delivery-delay early-warning engine for global health supply chains, built on IBM LinuxONE."
> 
> 

---

### **Slide 2: 01. The Challenge**

*(On-Screen: Problem Statement & Operational Context)*

> "For life-critical health deliveries, there is zero visibility when an order is placed. With 11.5 percent of shipments arriving late—spiking to 17.2 percent on RDC routes—patients face dangerous stockouts. Logistics officers currently rely on reactive manual tracking. We provide risk scoring at order time, turning a 128-day median lead time into actionable runway."
> 
> 

---

### **Slide 3: 02. Competitive Landscape**

*(On-Screen: How order-time scoring differs from tracking and planning tools)*

> "While tools like project44 and FourKites track shipments in transit via GPS, and SAP IBP focuses on macro planning, V12 Engines provides a targeted, per-order late-risk alert at order placement—giving officers maximum runway to act."
> 
> 

---

### **Slide 4: 03. Integrated Solution**

*(On-Screen: Order-Time Delay Warning Engine)*

> "Our solution uses a Snap ML boosted-tree classifier on IBM LinuxONE to rank delay risk. It features template-based driver explanations, secure SSH-encrypted tunnel access with audit logging, and an interactive dashboard for officer sign-off."
> 
> 

---

### **Slide 5: 04. Technical Architecture**

*(On-Screen: End-to-End System Pipeline on IBM LinuxONE)*

> "Our pipeline processes over 10,000 USAID SCMS shipment rows with rolling prior late rates. A native s390x Snap ML classifier scores risk, while FastAPI and Docker Compose power our real-time replay web frontend."
> 
> 

---

### **Video Walkthrough**

*(On-Screen: Dashboard, Model, and Platform Demo)*

> "In our live dashboard replay, orders are evaluated instantly at under 107 milliseconds latency. On our model tab, we achieve a PR-AUC of 0.28 against a 0.09 baseline on future test data, prioritizing high recall. And on our platform tab, everything runs securely on IBM LinuxONE containers via Docker Compose."
> 
> 

---

### **Slide 7: 05. Stakeholder Value**

*(On-Screen: Target Audience & Ecosystem Impact)*

> "This empowers logistics officers with early alerts, helps USAID and PEPFAR maintain HIV treatment continuity across 43 countries, and demonstrates secure, scalable AI inference directly on core enterprise infrastructure."
> 
> 

---

### **Slide 8: 06. Operational Impact**

*(On-Screen: Expected Outcomes & Model Metrics)*

> "Tested on post-cutoff data with a 9.5 percent late rate, our model catches 37.8 percent of late shipments at threshold. It cuts emergency air costs, pre-positions buffer stock, and ensures transparent supply chain governance."
> 
> 

---

### **Slide 9: 07. Key Takeaways**

*(On-Screen: Learnings During the Datathon)*

> "Key takeaways include deploying Snap ML and testing Granite on LinuxONE without dedicated GPUs, mastering leakage-safe temporal features, and executing rapid engineering.
> 
> 
> V12 Engines: Knowing before it’s late—every shipment, the moment it’s ordered."
> 
>
