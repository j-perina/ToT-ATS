"""
prompts.py

This module contains all prompt templates and category-specific Tree-of-Thought (ToT) execution chains
for evaluating resumes using OpenAI's API. Each category (experience, location, education, skills, etc.)
is processed through a 3-step prompt chain:

1. Extraction (e.g., E1, L1): Extract relevant resume content matching the job description.
2. Evaluation (e.g., E2, L2): Analyze the match between resume and job description.
3. Scoring (e.g., E3, L3): Assign a score (0-100) and a rationale.

The module includes:
- Prompt template functions for each stage.
- Execution functions (e.g., `run_experience_chain`) that call OpenAI and parse outputs.
- A centralized `call_openai` method.

Designed for use with ResumeScanner.ipynb, where input parsing, scoring orchestration, and result storage are handled.

"""

# Load API key from .env, Instantiate OpenAI client
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# --- OpenAI Call Function ---

def call_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


### Prompts for E's (Experience)

def E1_prompt(resume_experience, job_description):
    return f"""
You are a resume analysis system. Given the following resume's experience section and the job description for the job they are applying to, extract and summarize ONLY the relevant work experience fron the resume that matches the job description.
Be very picky with what you decide to keep. Do not be generous in favor of the applicant.
Here is the information from the resume and the job description for you to use.

Resume Experience:
{resume_experience}

Job Description:
{job_description}

Return just the relevant summarized experience as plain text.
"""

def E2_prompt(E1_output, job_description):
    return f"""
You are a professional resume reviewer working in HR for a highly competitive, selective, and presigous company, and your goal is to evaluate if a candidates job experience is relevant for a role your comapny is hiring for. 
Please compare the applicants summarized work experience and compare it to the requirments for the job description. 
Return a few sentences explaining how to candidate is qualified for the job based on their experience, and what they may be missing.

Applicants summarized expereince:
{E1_output}

Job Description:
{job_description}

Output:
"""

def E3_prompt(E2_output):
    return f"""
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how a candidates experience aligns with the role they are applying for, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence.

2. A one-sentence explaination of why the score was given, including the key rationale behind the given score

Input:
{E2_output}

Output format:
experience_score: <integer between 0 and 100>
experience_note: <one sentence note>
"""


# --- Prompt Chain Execution ---

def run_experience_chain(resume, job_description):
    # Step E1
    e1_prompt = E1_prompt(resume["experience"], job_description)
    E1_output = call_openai(e1_prompt)
    #print("\nE1 Output:\n", E1_output)

    # Step E2
    e2_prompt = E2_prompt(E1_output, job_description)
    E2_output = call_openai(e2_prompt)
    #print("\nE2 Output:\n", E2_output)

    # Step E3
    e3_prompt = E3_prompt(E2_output)
    E3_output = call_openai(e3_prompt)
    #print("\nE3 Output:\n", E3_output)

    # Extract score/note
    lines = E3_output.strip().split("\n")
    experience_score = int(lines[0].split(":")[1].strip())
    experience_note = lines[1].split(":", 1)[1].strip()

    return experience_score, experience_note



### Prompt for L's (Location)

def L1_prompt(resume_location, job_description):
    return f""" 
You are a resume analysis system. Given the following candidate's location from their resume and the job description for the job they are applying to, extract the candidate’s location and assess whether commuting to the job location is feasible (or if relocation might be required).
If there is no location given for the candidate on the resume, return that there is no location included for the location.
Here is the information from the resume and the job description for you to use.

Resume Location:
{resume_location}

Job Description:
{job_description}

Return just the location and commute feasibility analysis as plain text.
""" 

def L2_prompt(L1_output, job_description):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company, and your goal is to evaluate if a candidate's location is suitable for a role your company is hiring for. 
Please compare the candidate’s location (and commute feasibility) with the job’s location requirements. 
Return a few sentences explaining how the candidate’s location may impact their fit for the job, including any concerns or advantages. If no location for the candidate is given, note how they did not include the location, which is not professional.

Candidate's location and commute feasibility:
{L1_output}

Job Description:
{job_description}

Output:
""" 

def L3_prompt(L2_output):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how the candidate’s location aligns with the job’s location requirements, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence. Be very picky with how you grade them. Do not be generous in favor of the applicant.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score

Input:
{L2_output}

Output format:
location_score: <integer between 0 and 100>
location_note: <one sentence note>
""" 

# --- Prompt Chain Execution ---

def run_location_chain(resume, job_description):
    # Step L1
    l1_prompt = L1_prompt(resume["location"], job_description)
    L1_output = call_openai(l1_prompt)

    # Step L2
    l2_prompt = L2_prompt(L1_output, job_description)
    L2_output = call_openai(l2_prompt)

    # Step L3
    l3_prompt = L3_prompt(L2_output)
    L3_output = call_openai(l3_prompt)

    # Extract score/note
    lines = L3_output.strip().split("\n")
    location_score = int(lines[0].split(":")[1].strip())
    location_note = lines[1].split(":", 1)[1].strip()

    return location_score, location_note


### Prompt for ED's (Education)

def ED1_prompt(resume_education, job_description):
    return f""" 
You are a resume analysis system. Given the candidate's education section and the job description for the job they are applying to, extract and summarize only the relevant education credentials (such as degrees, majors, and certifications) from the resume that match the job description.
Here is the information from the resume and the job description for you to use.

Resume Education:
{resume_education}

Job Description:
{job_description}

Return just the relevant education details as plain text.
""" 

def ED2_prompt(ED1_output, job_description):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company, and your goal is to evaluate if a candidate's education is sufficient for the role your company is hiring for. 
Please compare the applicant’s summarized education background with the job’s education requirements. 
Return a few sentences explaining how the candidate’s education meets the job requirements and what they may be missing.

Candidate's summarized education:
{ED1_output}

Job Description:
{job_description}

Output:
""" 

def ED3_prompt(ED2_output):
    return f"""
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how a candidate’s education aligns with the job requirements, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence. DO not be afraid to rate them a poor fit if their education does not directly align with the job description.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score

Input:
{ED2_output}

Output format:
education_score: <integer between 0 and 100>
education_note: <one sentence note>
""" 

# --- Prompt Chain Execution ---

def run_education_chain(resume, job_description):
    # Step ED1
    ed1_prompt = ED1_prompt(resume["education"], job_description)
    ED1_output = call_openai(ed1_prompt)

    # Step ED2
    ed2_prompt = ED2_prompt(ED1_output, job_description)
    ED2_output = call_openai(ed2_prompt)

    # Step ED3
    ed3_prompt = ED3_prompt(ED2_output)
    ED3_output = call_openai(ed3_prompt)
    
    # Extract score/note
    lines = ED3_output.strip().split("\n")
    education_score = int(lines[0].split(":")[1].strip())
    education_note = lines[1].split(":", 1)[1].strip()
    return education_score, education_note

### Prompt for SK's (Skills)

def SK1_prompt(resume_skills, job_description):
    return f""" 
You are a resume analysis system. Given the candidate's listed skills from their resume and the job description for the job they are applying to, extract and list only the skills (including both hard and soft skills) from the resume that match the job description’s requirements.
Here is the information from the resume and the job description for you to use. Do not be generous for the applicant, do not "assume" they have a skill if they do not mention it.

Resume Skills:
{resume_skills}

Job Description:
{job_description}

Return just the relevant skills as plain text.
""" 

def SK2_prompt(SK1_output, job_description):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company, and your goal is to evaluate if a candidate's skills match the skills required for the role your company is hiring for. 
Please compare the candidate’s listed skills with the job’s required skills. 
Return a few sentences explaining how the candidate’s skills align with the job requirements and any important skills that might be missing. Be harsh.

Candidate's skills:
{SK1_output}

Job Description:
{job_description}

Output:
""" 

def SK3_prompt(SK2_output):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how a candidate’s skills align with the job requirements, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score, such as an important skill that they do not have.

Input:
{SK2_output}

Output format:
skills_score: <integer between 0 and 100>
skills_note: <one sentence note>
""" 

# --- Prompt Chain Execution ---

def run_skills_chain(resume, job_description):
    # Step SK1
    sk1_prompt = SK1_prompt(resume["skills"], job_description)
    SK1_output = call_openai(sk1_prompt)

    # Step SK2
    sk2_prompt = SK2_prompt(SK1_output, job_description)
    SK2_output = call_openai(sk2_prompt)

    # Step SK3
    sk3_prompt = SK3_prompt(SK2_output)
    SK3_output = call_openai(sk3_prompt)

    # Extract score/note
    lines = SK3_output.strip().split("\n")
    skills_score = int(lines[0].split(":")[1].strip())
    skills_note = lines[1].split(":", 1)[1].strip()
    
    return skills_score, skills_note

### Prompt for LA's (Languages)

def LA1_prompt(resume_text, job_description):
    return f""" 
You are a resume analysis system. Given the candidate's resume text and the job description for the job they are applying to, identify any languages (human languages like English, Spanish, etc.) that the candidate has listed as languages they speak. 
Here is the resume content and the job description for you to use.

Resume Text:
{resume_text}

Job Description:
{job_description}

Return just the list of languages spoken by the candidate as plain text.
""" 

def LA2_prompt(LA1_output, job_description):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company, and your goal is to evaluate if the languages the candidate speaks meet the requirements of the role. 
Please compare the languages the candidate knows with the languages required or preferred for the job. 
Return a few sentences explaining how the candidate’s language skills align with the job requirements and if any required languages are missing.

Candidate's languages:
{LA1_output}

Job Description:
{job_description}

Output:
""" 

def LA3_prompt(LA2_output):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how the candidate’s language skills align with the job requirements, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score

Input:
{LA2_output}

Output format:
languages_score: <integer between 0 and 100>
languages_note: <one sentence note>
""" 

# --- Prompt Chain Execution ---

def run_languages_chain(resume, job_description):
    # Step LA1
    resume_text = f"{resume.get('summary', '')}\n{resume.get('education', '')}\n{resume.get('experience', '')}\n{resume.get('skills', '')}"
    la1_prompt = LA1_prompt(resume_text, job_description)
    LA1_output = call_openai(la1_prompt)

    # Step LA2
    la2_prompt = LA2_prompt(LA1_output, job_description)
    LA2_output = call_openai(la2_prompt)

    # Step LA3
    la3_prompt = LA3_prompt(LA2_output)
    LA3_output = call_openai(la3_prompt)

    # Extract score/note
    lines = LA3_output.strip().split("\n")
    languages_score = int(lines[0].split(":")[1].strip())
    languages_note = lines[1].split(":", 1)[1].strip()
    
    return languages_score, languages_note


### Prompt for O's (Other Qualities)

def O1_prompt(resume_text, job_description):
    return f"""
You are a resume analysis system. Given the candidate's resume text and the job description for the job they are applying to, extract any mention of other relevant qualities or soft skills the candidate has (such as adaptability, familiarity with specific tools or methodologies, teamwork, communication skills, etc.) that would help them fit into the job’s environment or culture.
Here is the resume content and the job description for you to use.

Resume Text:
{resume_text}

Job Description:
{job_description}

Return just the relevant other attributes as plain text.
""" 

def O2_prompt(O1_output, job_description):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company, and your goal is to evaluate if the candidate's additional qualities (adaptability, soft skills, and other attributes) are relevant to the role. 
Please compare these attributes from the candidate with the job description’s desired qualities. 
Return a few sentences explaining how the candidate’s other attributes make them a good or bad fit for the job and if any important qualities are missing.

Candidate's other attributes:
{O1_output}

Job Description:
{job_description}

Output:
""" 

def O3_prompt(O2_output):
    return f""" 
You are a professional resume reviewer working in HR for a a highly competitive, selective, and presigous company. 
Based only on the following summary of how the candidate’s other attributes and soft skills align with the job requirements, assign:
1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence. Be harsh and do not be afraid to give a low score if inadequate other skills are displayed.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score

Input:
{O2_output}

Output format:
other_score: <integer between 0 and 100>
other_note: <one sentence note>
""" 

# --- Prompt Chain Execution ---

def run_other_chain(resume, job_description):
    # Step O1
    resume_text = f"{resume.get('summary', '')}\n{resume.get('education', '')}\n{resume.get('experience', '')}\n{resume.get('skills', '')}"
    o1_prompt = O1_prompt(resume_text, job_description)
    O1_output = call_openai(o1_prompt)

    # Step O2
    o2_prompt = O2_prompt(O1_output, job_description)
    O2_output = call_openai(o2_prompt)

    # Step O3
    o3_prompt = O3_prompt(O2_output)
    O3_output = call_openai(o3_prompt)

    # Extract score/note
    lines = O3_output.strip().split("\n")
    other_score = int(lines[0].split(":")[1].strip())
    other_note = lines[1].split(":", 1)[1].strip()

    return other_score, other_note


### Prompt for S's (Summary)

def S1_prompt(
    experience_score, experience_note,
    location_score, location_note,
    education_score, education_note,
    skills_score, skills_note,
    languages_score, languages_note,
    other_score, other_note
):
    return f"""
You are an advanced resume analysis system. You have just evaluated a candidate’s resume against a job description in six categories: experience, location, education, skills, languages, and other attributes.

Here are the evaluation results:

Experience: {experience_score}/100 — {experience_note}  
Location: {location_score}/100 — {location_note}  
Education: {education_score}/100 — {education_note}  
Skills: {skills_score}/100 — {skills_note}  
Languages: {languages_score}/100 — {languages_note}  
Other: {other_score}/100 — {other_note}

Based on these evaluations, summarize how well the candidate’s overall profile fits the expectations for this job.
Do not be afraid to say they are not fit for the job if they are lacking in some categories, as this is a prestigous company and we need to be very selective.
Return your summary as a concise paragraph of no more than 5 sentences.
"""

def S2_prompt(S1_output, job_description):
    return f"""
You are a professional resume reviewer in HR at a highly competitive, selective, and presigous company. Based on the following candidate evaluation summary and the job description, assess the candidate’s overall fitness for the role.
Be very strict and harsh. If they are missing an important feature, do not be leniant and make sure to communicate that they may not be fit for the job.

Candidate Evaluation Summary:
{S1_output}

Job Description:
{job_description}

Return a few sentences explaining the overall pros and cons of this candidate for the job’s requirements, identifying strengths and weaknesses.
"""

def S3_prompt(S2_output):
    return f"""
You are a professional HR resume reviewer at a highly competitive, selective, and presigous company. Based only on the following overall evaluation of a candidate, assign:

1. A score between 0 and 100 (integer only) where:
   - 90-100: Outstanding fit, top-tier candidate
   - 70-89: Strong fit, likely to be interviewed
   - 50-69: Somewhat qualified but significant weaknesses
   - 30-49: Poor fit for the role
   - 0-29: Unqualified or irrelevant background
Be precise and do not inflate the score without strong evidence. This summary will be used to rank applicants, so do not be afraid to dock points for weak sections of the application.
Be very picky with how you grade. Do not be generous in favor of the applicant. Don't be afraid to say grade someone well below average if they are lacking in an important part of their resume.

2. A one-sentence explanation of why the score was given, including the key rationale behind the given score

Evaluation:
{S2_output}

Output format:
summary_score: <integer between 0 and 100>  
summary_note: <one sentence note>
"""

# --- Prompt Chain Execution ---

def run_summary_chain(ats_row, job_description):
    # Step S1
    s1_prompt = S1_prompt(
        ats_row["experience_score"], ats_row["experience_note"],
        ats_row["location_score"], ats_row["location_note"],
        ats_row["education_score"], ats_row["education_note"],
        ats_row["skills_score"], ats_row["skills_note"],
        ats_row["languages_score"], ats_row["languages_note"],
        ats_row["other_score"], ats_row["other_note"]
    )
    S1_output = call_openai(s1_prompt)

    # Step S2
    s2_prompt = S2_prompt(S1_output, job_description)
    S2_output = call_openai(s2_prompt)

    # Step S3
    s3_prompt = S3_prompt(S2_output)
    S3_output = call_openai(s3_prompt)

    # Extract score and note
    lines = S3_output.strip().split("\n")
    summary_score = int(lines[0].split(":")[1].strip())
    summary_note = lines[1].split(":", 1)[1].strip()
    
    return summary_score, summary_note


### One Shot Prompts

def run_oneshot_chain(resume, job_description, model="gpt-4o"):
    """
    One-shot prompt that sends the full resume and job description to the LLM.
    Returns all scores and notes, with the summary_score computed last based on the others.
    """

    resume_text = f"""
    Name: {resume['name']}
    Location: {resume['location']}
    Summary: {resume['summary']}
    Education: {resume['education']}
    Experience: {resume['experience']}
    Skills: {resume['skills']}
    """

    prompt = f"""
You are an experienced HR resume reviewer. Given the full resume and job description below, evaluate the candidate in the following categories.

For each category, assign a score from 0 to 100 where:
- 90–100: Excellent match
- 70–89: Strong match
- 50–69: Moderate match
- 30–49: Weak match
- 0–29: Poor or no match

Include a brief justification (1–2 sentences) with each score.

The categories are:
- location
- experience
- education
- skills
- languages
- other

Once those are complete, compute an overall **summary_score** (0–100) and **summary_note** that reflects how well the candidate matches the job, based on the above categories.

Resume:
{resume_text}

Job Description:
{job_description}

Output format:
location_score: <int>
location_note: <short explanation>

experience_score: <int>
experience_note: <short explanation>

education_score: <int>
education_note: <short explanation>

skills_score: <int>
skills_note: <short explanation>

languages_score: <int>
languages_note: <short explanation>

other_score: <int>
other_note: <short explanation>

summary_score: <int>
summary_note: <short explanation>
"""

    response = call_openai(prompt, model=model)
    return response


def parse_oneshot_response(response):
    """
    Parses the response from run_oneshot_chain() into a dictionary 
    with keys matching the ats_results DataFrame columns.
    """
    fields = [
        "location_score", "location_note",
        "experience_score", "experience_note",
        "education_score", "education_note",
        "skills_score", "skills_note",
        "languages_score", "languages_note",
        "other_score", "other_note",
        "summary_score", "summary_note"
    ]

    result = {}
    lines = response.strip().splitlines()

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if "score" in key:
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = None
            elif "note" in key:
                result[key] = value

    # Fill in missing fields as None
    for field in fields:
        if field not in result:
            result[field] = None

    return result
