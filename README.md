Here’s a polished English README.md following GitHub best practices for your Research-agent-main project:


---

Research Agent Main 🧠

A modular, autonomous research agent powered by LLMs. It automates multi-stage online and local research workflows—breaking down complex questions, gathering information, refining results, and synthesizing a well-structured final report.


---

Table of Contents

Overview

Features

Demo Screenshots

Getting Started

Prerequisites

Installation

Configuration


Usage

Command-Line Interface

Python API


Project Structure

Development & Testing

Contributing

License

Acknowledgments



---

Overview

Research Agent Main enables you to create research tasks that:

1. Plan by decomposing a main topic into sub‐questions.


2. Execute searches (web or local documents) to gather relevant data.


3. Iterate by analyzing results and refining follow-up queries.


4. Synthesize findings into an organized, citation-rich report.



This pipeline delivers multi-hop factual research with minimal manual intervention.


---

Features

🧭 Task planner: Generates strategic sub-questions for coverage depth

🔍 Parallel search execution: Efficient browsing, scraping, and content extraction

🔄 Iterative refinement: Recognizes knowledge gaps and auto-adjusts research queries

📝 Report synthesis: Structured Markdown output with executive summary, sections, and sources

🧠 Multi-agent coordination: Planner, crawler, analyzer, and synthesizer working together

📂 Supports web + local files: PDF, TXT, markdown, etc.

⚙️ Extensible modular design: Easy to customize or add new agents/tools



---

Demo Screenshots

(Insert screenshots or short terminal clips of agent planning, crawling, and final output.)


---

Getting Started

Prerequisites

Python 3.10+

.env with LLM API keys (e.g., OpenAI, Gemini) and search tools (e.g., Serper, DuckDuckGo)


Installation

git clone https://github.com/Abokabera2024/Research-agent-main.git
cd Research-agent-main
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

Configuration

Create a .env file (copy from .env.example) with entries like:

OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_key
DOC_PATH=./local_docs   # Optional: for local-document research


---

Usage

Command-Line Interface

python main.py \
  --query "Impacts of quantum computing on cybersecurity" \
  --depth 2 \
  --max_iterations 4 \
  --output report.md

--query: primary research question

--depth: levels of sub-questions

--max_iterations: refinement cycles

--output: report file path


Python API

from research_agent import ResearchAgent

agent = ResearchAgent(
    query="Climate change effects on agriculture",
    depth=3,
    max_iterations=3
)

report = agent.run()
print(report)  # Markdown output


---

Project Structure

.
├── .env.example            # Required API keys & config
├── main.py                 # CLI entry point
├── research_agent/         # Core agent modules
│   ├── planner.py          # Planner agent: creates sub-questions
│   ├── search.py           # Search agent: runs web/local lookups
│   ├── analyzer.py         # Analyzer agent: extracts key insights
│   ├── synthesizer.py      # Synthesizer: builds final report
│   └── utils.py            # Common utilities
├── local_docs/             # Place documents here for local research
├── requirements.txt        # Python dependencies
└── README.md               # This file


---

Development & Testing

Unit tests: pytest tests/

Style & formatting: flake8 . & black .

Add new tools/agents: follow modular structure in research_agent/



---

Contributing

We welcome contributions! To get started:

1. Fork the repo


2. Create a feature branch (git checkout -b feature/xyz)


3. Write tests + documentation


4. Open a pull request



Please follow the existing project style and include docs for any new feature.


---

License

This project is licensed under the MIT License — see the LICENSE file for details.


---

Acknowledgments

Inspired by multi-agent research frameworks like GPT-Researcher and Open Deep Research

Thank you to all upstream contributors in the autonomous research agent community


