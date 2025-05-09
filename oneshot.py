from prompts import run_oneshot_chain, parse_oneshot_response
import pandas as pd

def evaluate_all_oneshot_resumes(resumes, job_description):
    """
    Loops through resumes and evaluates each one using the one-shot prompt approach.
    Returns a DataFrame identical in structure to ats_results.
    """
    oneshot_columns = [
        "id",
        "summary_score", "summary_note",
        "location_score", "location_note",
        "experience_score", "experience_note",
        "education_score", "education_note",
        "skills_score", "skills_note",
        "languages_score", "languages_note",
        "other_score", "other_note",
        "composite_score"
    ]
    
    df = pd.DataFrame(columns=oneshot_columns)

    for i, resume in enumerate(resumes):
        print(f"Running one-shot evaluation for resume {i + 1}...")

        # Run one-shot LLM call and parse result
        response = run_oneshot_chain(resume, job_description)
        parsed = parse_oneshot_response(response)

        # Compute composite score using standard weights
        # Safe fallback using get() and default to 0 if value is None
        composite_score = (
            0.3 * (parsed.get("experience_score") or 0) +
            0.2 * (parsed.get("skills_score") or 0) +
            0.2 * (parsed.get("education_score") or 0) +
            0.1 * (parsed.get("languages_score") or 0) +
            0.1 * (parsed.get("other_score") or 0) +
            0.1 * (parsed.get("location_score") or 0)
        )


        df.loc[i] = {
            "id": i + 1,
            **parsed,
            "composite_score": round(composite_score, 2)
        }

    return df
