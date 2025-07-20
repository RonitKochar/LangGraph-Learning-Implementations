Below is an overview of each of the six agent and RAG projects included in this repository. These descriptions are ready to paste directly into your README prior to pushing to GitHub.

1. Poetry
Poetry is a modular AI assistant designed using Python to answer user queries by blending LLM reasoning, real-time search, and programmable logic flows.

Key Technologies
LangGraph: Dynamic, stateful reasoning graph orchestration.

ReAct (Reason + Act): Planning with alternation between reasoning and tool use.

Tavily Search: Real-time web search integration for up-to-date, factual answers.

LangChain LLM: Natural language generation and logic.

Architecture & Features
StateGraph Workflow: State machine design with looping over reasoning/tool-use nodes for iterative problem-solving.

Custom Nodes: Modular, extensible, and configurable nodes that interpret user intent, invoke search, post-process data, and handle general queries.

Intelligent Query Handling: Detects when to search, calculates on extracted information, and defers to LLM as needed.

Separation of Roles: Clean file organization (workflow, reasoning, tools).

Environment Configuration: Secure API management via environment variables.

2. Reflection Agent
The Reflection Agent is an iterative generator and refiner for short-form content (e.g., tweets), simulating expert influencer critique and editorial improvement.

Key Features
Generate-Reflect Cycle: Alternates between content creation and critical feedback.

Prompt Engineering:

Generation Prompt for crafting and revising posts.

Reflection Prompt for expert-style quality grading and actionable suggestions.

Stateful Graph Workflow: Each step tracked by a stateful message-passing graph, ensuring context-aware improvement.

Conditional Logic: Loop continues until content quality criteria are met (by iteration cap or sufficiency).

Full Transparency: Tracks and refines every message for reproducibility.

3. Reflexion Agent
Reflexion Agent elevates automated research by generating, self-critiquing, and refining in-depth answers to research questions.

Key Features
Iterative Answer Refinement: Each answer (~250 words) is critically reviewed for both completeness and brevity; shortcomings are addressed via suggested web queries.

Self-Critique: Pinpoints missing and superfluous information in responses, guiding targeted improvement.

Integrated Search Tools: Automates query creation and execution to gather new evidence for answer upgrades.

Strict Schemas: Shapes answers, reflections, search intent, and (on revision) references into cleanly structured, testable outputs.

Directed Graph Workflow: Orchestrates draft, search, critique, and revision cycles with configurable depth.

Prompt Engineering: Ensures rigorous, expert-grade research conduct at each turn.

4. Agentic RAG
Agentic RAG is a modular pipeline for Retrieve-Augment-Generate (RAG) systems, combining semantic search, iterative vetting, and grounded answer generation.

Key Features
Vector Search & Retrieval:

Ingests and embeds web documents for semantic chunk-based search.

Retrieves high-relevance context efficiently from a persistent vector store.

Automated Relevance Grading:

Binary LLM grader ranks and filters contexts, ensuring only the most pertinent information is used.

Fallback Web Search:

Supplements missing or outdated vectorstore content with live web results.

LLM-Augmented Generation:

Final answers synthesized by an LLM, using only rigorously filtered and up-to-date content as context.

Graph-Based Orchestration:

Modular, conditional state graph from retrieval to grading to answer production, ensuring easy extension and clarity.

5. Self RAG
Self RAG builds on the Agentic RAG foundation by introducing automated, model-based self-evaluation and correction for robust, fact-grounded responses.

Key Innovations
Hallucination Detection: Every answer is checked for factual grounding against retrieved evidence; unsupported statements are flagged and rectified.

Dual Graders:

Grounding Grader ensures answers are strictly based on given documents.

Answer Grader checks for direct, full resolution of the initial user question.

Autonomous Self-Improvement:

If an answer is unsupported or off-target, the system cycles through retrieval, revision, and regeneration until standards are met.

Iterative, Evidence-Grounded Correction:

Repeats vetting and correction until output is both relevant and trustworthy.

Testable Modular Chain:

Full coverage in tests for all grading and fix-up pathways.

6. Adaptive RAG
Adaptive RAG is the most dynamic in this suite, introducing query routing and source adaptation based on question content.

Key Features
Intelligent Query Router:

An LLM-based router analyzes every question, sending in-domain topics (agents, prompt engineering, adversarial attacks) to the vector database, and everything else to web search.

Adaptive Retrieval:

Ensures the best knowledge source for every user request—domain-specific or open-domain.

Full Self-Evaluation Loop:

Rigorous hallucination/answer grading cycles, as in Self RAG, with dynamic source reassignment when initial iterations fall short.

Flexible, Inspectable State Graph:

Clearly organized control flow for easy debugging and extensibility as knowledge/criteria evolve.