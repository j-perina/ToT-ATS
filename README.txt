ToT-ATS
=======================

Overview
--------
This project implements an AI-powered resume evaluation system using OpenAI’s GPT models.
It compares candidate resumes to a job description using two different prompting strategies:

1. **Tree-of-Thought Prompting (ToT)**: 
   Each resume is evaluated across six categories (Experience, Location, Education, Skills, Languages, Other), using a three-step reasoning chain per category. A final summary score is then computed based on those categories.

2. **One-Shot Prompting**:
   The resume is evaluated in a single LLM prompt where all categories are scored at once.

The system also compares its internal rankings with a benchmark ranking from an external ATS system provided in `ATS_Website_Results.xlsx`.

Key Features
------------
- Modular prompting chains for structured evaluation
- Composite and summary scoring for ranking resumes
- Head-to-head comparison of ToT vs One-Shot vs Website ATS
- Visualization tools for correlation, scoring, and rank differences
- Optional multi-run averaging for increased result robustness

Required Files
--------------
- `resume_scanner.ipynb` — Main notebook to run both ToT and One-Shot evaluations
- `main_config.py` — Stores job description and file paths
- `prompts.py` — Prompt logic and LLM chains for Tree-of-Thought and One-Shot evaluations
- `oneshot.py` — Executes and parses one-shot evaluations
- `data_loader.py` — Handles loading resumes and caching logic
- `analysis.py` — Provides utilities for ranking, plotting, and comparing results
- `resumes.xlsx` — Input file containing resume data
- `ATS_Website_Results.xlsx` — External ATS rankings used for comparison

Environment Setup
-----------------
This project requires access to the OpenAI API. You must create a `.env` file in the project root directory with the following line:
OPENAI_API_KEY=your-openai-api-key-here

No additional packages beyond standard Python libraries (`pandas`, `openpyxl`, `matplotlib`, `seaborn`, `scipy`, etc.) are required.

How It Works
------------
1. **Resume Loading**:
   Resumes are loaded from `resumes.xlsx` using `data_loader.py` and stored as dictionaries.

2. **Evaluation**:
   - The Tree-of-Thought mode uses a sequence of LLM calls per resume category.
   - The One-Shot mode sends the full resume in a single LLM call.
   - Both generate a `summary_score` and optional `composite_score`.

3. **Caching**:
   Cached results can be loaded to avoid re-running the model. Set `force_rerun=True` to regenerate results.

4. **Analysis**:
   - `analysis.py` provides plotting tools to visualize score trends and rank differences.
   - `merge_all_ranks()` and `compare_ranks()` help assess alignment between models.

5. **Multi-Run Averaging (optional)**:
   To reduce LLM variability, you can run the ToT model multiple times using a helper function (`run_tot_multiple_times`) and average the results. The output matches the same schema as a normal ToT run.

Output Files
------------
- `ATS_Results.xlsx`: Main Tree-of-Thought output
- `ATS_Results_Stored.xlsx`: Cached ToT output
- `ATS_Oneshot_Results.xlsx`: Main One-Shot output
- `Oneshot_Results_Stored.xlsx`: Cached One-Shot output

Authors
------
James Perina  
Victor Rouzer
Amr Eldessouky
Rami Ghaleb
