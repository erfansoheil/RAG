# 🧭 Project Roadmap: Retrieval & Agentic AI in Finance -
**Author:** Erfan SOHEIL
**Last Updated:** October 2025  

## Overview
This repository explores **retrieval strategies in LangChain** through a series of progressively complex Jupyter notebooks — moving from conceptual demonstrations to practical, semi‑agentic systems in the **finance domain**.

The overall goal is to **teach and demonstrate real‑world Retrieval‑Augmented Generation (RAG) and Agentic AI** workflows using accessible, open financial data.

---

## Notebook 1 – Comparative Retrieval Study
**Title:** _Side‑by‑Side Comparison of Retrieval Methods in Financial QA_

**Goal:** Evaluate and compare five retrieval approaches — Basic, MMR, Parent Document, Multi‑Query, and Self‑Query — on a realistic dataset (e.g., SEC 10‑K filings).  

**What it does:**
- Loads and preprocesses financial text data (long 10‑K sections).  
- Runs each retrieval method on identical queries.  
- Measures relevance, completeness, and diversity of retrieved chunks.  
- Highlights where basic retrieval fails and how advanced techniques improve results.

**Outcome:** A clear, data‑driven comparison showing that combining multiple retrieval strategies yields superior financial insights.

---

## Notebook 2 – Pseudo‑App Demonstration
**Title:** _Building a Financial QA Web App with LangChain Retrieval_

**Goal:** Turn Notebook 1’s findings into an interactive, working prototype.  

**What it does:**
- Creates a lightweight **Streamlit or Gradio web app** (no Google Auth, no heavy backend).  
- Lets users ask questions about company reports.  
- Displays answers, retrieved sources, and retrieval method comparisons side‑by‑side.  
- Focuses on clean engineering, modular structure, and usability.

**Outcome:** A functional demo app proving how retrieval architectures translate to real products — useful for both learning and showcasing engineering capability.

---

## Notebook 3 – Semi‑Agentic Financial Assistant
**Title:** _From Retrieval to Agency: Dynamic Query Routing in Finance_

**Goal:** Extend Notebook 1’s retrieval logic into a **semi‑agentic pipeline** that autonomously selects the best retrieval strategy based on query type.  

**What it does:**
- Introduces a simple decision layer (rule‑based or LLM‑based) to choose retrieval mode.  
- Chains reasoning + retrieval + generation steps.  
- Demonstrates a mini agent that can adapt to different financial information needs (e.g., numeric summaries vs. qualitative insights).  

**Outcome:** A flexible, partially autonomous financial assistant — bridging RAG with early‑stage Agentic AI.

---

## Final Deliverable
By the end of this series, readers and learners will have:
1. **A deep understanding of retrieval strategies** and when to use each.  
2. **A runnable financial QA app** demonstrating real engineering practices.  
3. **A semi‑agentic retrieval pipeline**, showing the evolution from static RAG to adaptive, intelligent systems.

This roadmap represents a complete learning and showcase journey — from retrieval fundamentals to applied Agentic AI in finance.


