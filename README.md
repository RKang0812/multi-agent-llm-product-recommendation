# Multi-Agent LLM Application for Product Recommendation

A structured LLM application demonstrating multi-agent orchestration, tool-augmented reasoning, and multimodal processing.

This project explores how large language models can be organized into specialized agents to collaboratively solve real-world product recommendation tasks.


## Overview

This application implements a task-oriented multi-agent workflow for product recommendation and product analysis.

It supports:
- Text-based product recommendation
- Image-based product understanding (Vision model)
- Real-time web search for pricing and availability
- Region-aware output formatting (EUR / CNY)
- The project focuses on system design rather than UI complexity.



## Architecture

The system is built around CrewAI-based agent orchestration.

### Agents

1. **Recommendation Agent**
- Parses user requirements
- Performs structured product reasoning
- Generates ranked product suggestions

2. **Vision Agent**
- Uses GPT-4o Vision for image understanding
- Extracts product attributes
- Answers product-related questions from images

3. **Web Search Agent**
- Integrates Serper API
- Retrieves up-to-date pricing and availability
- Provides source-aware responses



## Workflow

```
User Input
   ↓
Streamlit Interface
   ↓
CrewAI Orchestrator
   ↓
Agent Task Delegation
   ↓
Tool Usage (OpenAI API + Web Search)
   ↓
Response Aggregation
   ↓
Structured Output
```


## Technical Stack

- Python 3.8+
- OpenAI GPT-4o / GPT-4o-mini
- CrewAI (multi-agent orchestration)
- LangChain (LLM abstraction layer)
- Streamlit (UI layer)
- Serper API (real-time web search)



## Engineering Highlights

- Modular agent architecture (independently executable agents)
- Separation of orchestration and UI logic
- Tool-augmented LLM reasoning
- Multimodal input handling (text + image)
- Environment-based configuration management
- Cost-aware model selection strategy
- Region-aware response formatting


## Set Up

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Configure Environment Variables

Create a .env file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

### Run

```bash
streamlit run main.py
```

## Project Structure

```
Product-Recommendation-System/
│
├── main.py                  # Main application with all three agents
├── recommend_agent.py       # Standalone product recommendation agent
├── image_agent.py           # Standalone image analysis agent
├── web_search_agent.py      # Standalone web search agent
│
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore rules
│
└── README.md               # This file
```

## Architecture

```
User Input → Streamlit Interface
                ↓
         CrewAI Orchestrator
                ↓
        ┌───────┴───────┐
        ↓               ↓
   AI Agents      OpenAI API
        ↓               ↓
   Web Search    GPT-4/GPT-4o-mini
        ↓               ↓
   Product Info   Analysis Results
        ↓               ↓
        └───────┬───────┘
                ↓
         Final Response
```

### Multi-Agent Workflow

1. **Query Analysis**: System determines which agent(s) to use
2. **Agent Execution**: Relevant agents work on their specific tasks
3. **Web Search**: Agents use Serper API to fetch current information
4. **LLM Processing**: OpenAI models analyze and generate responses
5. **Result Synthesis**: Agents combine findings into comprehensive answer
6. **User Display**: Formatted results shown in Streamlit interface
