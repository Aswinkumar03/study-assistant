# Study Assistant
 
A local RAG (retrieval-augmented generation) chatbot that answers questions about my lecture notes and shows which slides each answer came from. Everything runs on an RTX 5070ti using Ollama, so no lecture material leaves my machine.
 
## Highlights
- Answers questions from PDF lecture slides, with source file and page citations
- Understands follow-up questions ("what about logistic regression?") by rewriting them into standalone questions
- Says "I don't know" when the answer is not in the notes
- Evaluated on 30 questions across 2 lecture decks: 85% baseline, 95 to 100% after tuning, 10/10 on a held-out set (see [Evaluation](#evaluation))
## How it works
1. PDFs in `notes/` are split into chunks (about 300 words, with overlap), keeping the page number of each chunk
2. Each chunk is embedded with `nomic-embed-text` and stored in a Chroma vector database
3. Your question is rewritten into a standalone question using the chat history
4. The closest chunks are retrieved (`TOP_K = 16`) and sent to `llama3.1:8b` together with the question
5. The model answers using only that context, and the sources are shown
## Tech stack
Python 3.12, Ollama (`llama3.1:8b`, `nomic-embed-text`), ChromaDB, pypdf
 
## Project structure
```
study-assistant/
  src/rag.py                    # ingestion, retrieval and chat loop
  eval/questions.json           # 20 tuning questions
  eval/questions_heldout.json   # 10 held-out questions (not used for tuning)
  eval/run_eval.py              # scoring script
  eval/results.csv              # score history for every run
  notes/                        # your own PDFs (not included in this repo)
  requirements.txt
```
 
## Setup
1. Install Ollama, then run:
   - `ollama pull llama3.1:8b`
   - `ollama pull nomic-embed-text`
2. Create a virtual environment and install packages:
   - `python -m venv .venv`
   - `.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`
3. Put your own PDFs in a `notes/` folder (text-based PDFs, not scanned images)
4. Build the database: `python src/rag.py ingest`
5. Ask questions: `python src/rag.py`
### Setup problems I ran into
- **`pip install chromadb` fails while building numpy:** this happens on very new Python versions that have no prebuilt packages yet. Use Python 3.12.
- **PowerShell blocks `Activate.ps1`:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
## Challenges and how I fixed them
I tested the assistant on real lecture questions and fixed what broke. Each fix was checked against the evaluation set below.
 
### 1. List questions were incomplete
**Problem:** I asked "give me all the algorithms that are used for classification". The first version only returned Random Forest and Support Vector Machines, even though the slides cover six classification algorithms.
 
**Cause:** the model can only use what retrieval gives it. With `TOP_K = 4`, only four chunks reached the model, and the other algorithms were on slides that did not make the cut.
 
**Fix:** I raised `TOP_K` from 4 to 8 and finally to 16, which returned four algorithms and then more. Because a bigger prompt needs a bigger context window, I also set `num_ctx` explicitly, since Ollama silently cuts off the start of a prompt that is too long. Finally I changed the system prompt to tell the model to include every relevant item that appears anywhere in the context. After that, the full list question passed in the final runs.
 
### 2. Follow-up questions had no memory
**Problem:** after the classification question, I asked "wht abt logistic regression". The answer did not address it properly.
 
**Cause:** the script sent only the current question to the model, and the context window only holds what is sent in each request. The model never saw the earlier question, and retrieval searched using just four words, so it pulled weak chunks.
 
**Fix:** I keep a short chat history and, before retrieving, ask the model to rewrite the new question as a standalone one (for example, "What is logistic regression and is it used for classification?"). Retrieval then searches with the full question. The follow-up "what about random forest and logistic regression" now returns a correct answer for both algorithms, with page references.
 
### 3. Source lists were too long
**Problem:** answers listed up to 8 source pages, which made them hard to check.
 
**Fix:** Chroma returns chunks sorted from closest to furthest, so I now show only the first 3 distinct pages.
 
### 4. Scores changed from run to run
**Problem:** the same code scored anywhere from 17 to 20 out of 20 across 17 runs, so I could not tell whether a change helped.
 
**Cause:** the model samples its words with some randomness.
 
**Fix:** I set `temperature` to 0 with a fixed `seed`. Scores are much steadier (19 and 20 out of 20 in two runs), though GPU maths can still cause a small difference.
 
### 5. Avoiding fooling myself
**Problem:** I tuned the prompt while looking at the same 20 questions, so a high score there could just mean it learned that test.
 
**Fix:** I wrote 10 extra questions in `eval/questions_heldout.json` and ran them once, without changing anything afterwards. I also report the average over repeated runs instead of only the best run.
 
## Evaluation
`eval/run_eval.py` runs a set of questions through the assistant and scores each one:
- **Answer check:** all expected keywords must appear in the answer (for unanswerable questions, the answer must say it doesn't know)
- **Source check:** at least one retrieved chunk must come from the correct lecture and page
The tuning set has 20 questions (18 answerable, 2 not in the notes). The held-out set has 10 questions (9 answerable, 1 not in the notes).
 
| Version | Answer accuracy | Source retrieval |
|---|---|---|
| Baseline | 17/20 (85%) | 18/18 |
| After tuning, default settings (17 runs) | about 89% average, range 17 to 20 out of 20 | 18/18 |
| After tuning, temperature 0 (2 runs) | 19/20 and 20/20 | 18/18 |
| Held-out questions | 10/10 | 9/9 |
 
Run it yourself:
```
python eval/run_eval.py my-label
python eval/run_eval.py heldout eval/questions_heldout.json
```
 
### Limitations
- The test sets are small, so a single question changes the score by 5 percentage points (3 for the held-out set). Treat the numbers as indicative, not exact.
- The answer check looks for keywords, so an answer with the right words but a wrong claim could pass. I checked a sample of answers by hand.
- Source retrieval was already 100% at baseline. The remaining mistakes came from how the model worded answers, not from missing chunks.
- The held-out questions are mostly single-fact lookups, which are easier than list questions.
- It only works with text-based PDFs, and answers are limited by the 8B model.
## Future work
- Web chat page with login so I can use it from my phone over Tailscale
- Quiz mode that generates practice questions from a chosen week
- A larger evaluation set with answers checked by hand