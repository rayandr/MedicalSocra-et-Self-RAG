# MedicalSocra-et-Self-RAG


**Self-RAG (diabetes Q&A)**: routes questions to a local vector DB or web search, then self-checks the answer.


**Medical Socratic RAG**: simulates a short debate between two “doctors”, optionally grounded in local medical docs, and scored with a CRIT evaluator.



### Quickstart

- **Install dependencies**

```bash
pip install -r requirements.txt
```


- **Set API keys:**

  - For **Self-RAG**: `GROQ_API_KEY` and `TAVILY_API_KEY`
  - For **Medical** (and `crit.py`): `OPENAI_API_KEY_STAGE`

### Files

- `self_rag_v6.py`: Self-RAG pipeline (diabetes-focused sources + web fallback).
- `chat_interface_test.py`: a simple Gradio chat UI for Self-RAG (**currently imports** `self_rag_v6.py`).
- `medical_socra_RAG_v3.py`: Medical Socratic RAG workflow (two perspectives + optional retrieval + CRIT scoring).
- `crit.py`: CRIT scorer used by the medical workflow.
- `app.py`: Streamlit UI for the medical assistant (**currently imports** `medical_socra_RAG_v3_copy.py`).

### Run


- **Self-RAG chatbot (Gradio UI)**

```bash
python chat_interface_test.py
```


- **Medical assistant (Streamlit UI)**

```bash 
streamlit run app.py
```
