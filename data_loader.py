# data_loader.py

import os
import pandas as pd
from prompts import (
    run_experience_chain,
    run_location_chain,
    run_education_chain,
    run_skills_chain,
    run_languages_chain,
    run_other_chain,
    run_summary_chain
)
from main_config import ONESHOT_CACHED_RESULTS_PATH

def load_resumes(path="resumes.xlsx"):
    """
    Reads and parses resume data from an Excel file into a structured format.
    Each resume is converted into a dictionary containing key sections like name, location, summary, education, experience, and skills.
    Returns a list of these resume dictionaries for further processing.
    """
    df = pd.read_excel(path)
    resumes = []
    for _, row in df.iterrows():
        resume = {
            "name": row["name"],
            "location": row["location"],
            "summary": row["summary"],
            "education": row["education"],
            "experience": row["experience"],
            "skills": row["skills"]
        }
        resumes.append(resume)
    return resumes



def load_or_generate_ats_results(resumes, job_description, load_path="ATS_Results_Stored.xlsx", save_path="ATS_Results.xlsx", force_rerun=False):
    """
    Manages the ATS (Applicant Tracking System) evaluation process with caching capabilities.
    Either loads existing evaluation results from cache or performs a full evaluation of resumes against a job description.
    Implements a weighted scoring system across multiple resume components and saves results for future use.

    Args:
        resumes (list[dict]): Parsed resume dictionaries.
        job_description (str): Job description string.
        load_path (str): File to load cached results from.
        save_path (str): File to save new results to after rerun.
        force_rerun (bool): If True, always recompute; otherwise load if exists.

    Returns:
        pd.DataFrame: Full ATS results.
    """
    if not force_rerun and os.path.exists(load_path):
        ats_results = pd.read_excel(load_path)
        print(f"[INFO] Loaded cached ATS results from {load_path}")
        return ats_results

    print("No valid cached ATS results found or force_rerun=True. Running full ToT evaluation...")

    # Normal ToT Logic Loop
    ats_columns = [
    "id",
    "summary_score", "summary_note",
    "location_score", "location_note",
    "experience_score", "experience_note",
    "education_score", "education_note",
    "skills_score", "skills_note",
    "languages_score", "languages_note",
    "other_score", "other_note"]
    ats_results = pd.DataFrame(columns=ats_columns)

    for i, resume in enumerate(resumes):

        # Run ToT scoring chains
        experience_score, experience_note = run_experience_chain(resume, job_description)
        location_score, location_note = run_location_chain(resume, job_description)
        education_score, education_note = run_education_chain(resume, job_description)
        skills_score, skills_note = run_skills_chain(resume, job_description)
        languages_score, languages_note = run_languages_chain(resume, job_description)
        other_score, other_note = run_other_chain(resume, job_description)

        # Store partial results
        ats_results.loc[i] = {
            "id": i + 1,
            "summary_score": None,
            "summary_note": None,
            "location_score": location_score,
            "location_note": location_note,
            "experience_score": experience_score,
            "experience_note": experience_note,
            "education_score": education_score,
            "education_note": education_note,
            "skills_score": skills_score,
            "skills_note": skills_note,
            "languages_score": languages_score,
            "languages_note": languages_note,
            "other_score": other_score,
            "other_note": other_note
        }

        # Compute summary score
        summary_score, summary_note = run_summary_chain(ats_results.loc[i], job_description)
        ats_results.loc[i, "summary_score"] = summary_score
        ats_results.loc[i, "summary_note"] = summary_note

        # Print checkpoint summary
        print(f" Evaluated resume #{i + 1} - Experience Score: {experience_score} & Summary Score: {summary_score}")

    # Compute composite score before saving for tie-breakers
    ats_results["composite_score"] = (
    0.3 * ats_results["experience_score"] +
    0.2 * ats_results["skills_score"] +
    0.2 * ats_results["education_score"] +
    0.1 * ats_results["languages_score"] +
    0.1 * ats_results["other_score"] +
    0.1 * ats_results["location_score"]
    )

    ats_results.to_excel(save_path, index=False)
    print(f"[INFO] New ATS results saved to {save_path}")
    return ats_results

def run_or_load_oneshot_evaluation(resumes, job_description, use_cache=True):
    """
    Handles one-shot evaluation of resumes against a job description with caching support.
    Either retrieves previously cached results or performs a new evaluation if no cache exists.
    Saves results to both a cache file and an active results file for immediate use.

    Parameters:
        resumes (list): List of resume dictionaries.
        job_description (str): The job description text.
        use_cache (bool): If True, try to load cached results from disk.

    Returns:
        pd.DataFrame: The one-shot results.
    """
    if use_cache and os.path.exists(ONESHOT_CACHED_RESULTS_PATH):
        print(f"[INFO] Loaded cached One-Shot results from {ONESHOT_CACHED_RESULTS_PATH}")
        return pd.read_excel(ONESHOT_CACHED_RESULTS_PATH)

    print("[INFO] Running One-Shot evaluation for all resumes...")
    oneshot_results = evaluate_all_oneshot_resumes(resumes, job_description)

    # Save to both cache and active use path
    oneshot_results.to_excel(ONESHOT_CACHED_RESULTS_PATH, index=False)
    oneshot_results.to_excel("ATS_Oneshot_Results.xlsx", index=False)
    print(f"[INFO] One-Shot results saved to {ONESHOT_CACHED_RESULTS_PATH} and ATS_Oneshot_Results.xlsx")

    return oneshot_results