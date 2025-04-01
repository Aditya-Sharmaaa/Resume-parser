from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
import docx
from pathlib import Path
import re
import os
import json
from openai import OpenAI 
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import concurrent.futures
import groq
import psycopg2
from psycopg2.extras import execute_batch
from typing import Dict, List, Optional, Union
import logging
import uuid

# Load environment variables
env_path = find_dotenv('D:/lifeline/api.env')
load_dotenv(env_path)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )


def extract_text(file_path: Path) -> str:
    """Extract text from PDF, DOCX, or TXT files."""
    try:
        text = ""
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            raise ValueError("Unsupported file format. Only PDF, DOCX, and TXT are supported.")

        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from {file_path.name}: {str(e)}")

def parse_resume_with_llama(text: str) -> dict:
    """Parse resume text using Groq's Llama to match all 14 tables."""
    try:
        prompt = f"""
You are an expert resume parser. Extract all details from the resume text to perfectly match our 14-table database schema. 
Return the data in this exact JSON structure with all sections:

{{
    "basic_info": {{
        "photo": "URL or null",
        "salutation": "string or null",
        "first_name": "string",
        "middle_name": "string or null",
        "last_name": "string",
        "email": "string",
        "secondary_email": "string or null",
        "mobile": "string",
        "secondary_mobile": "string or null"
    }},
    "address": {{
        "address_line_1": "string or null",
        "address_line_2": "string or null",
        "city": "string or null",
        "state": "string or null",
        "country": "string or null",
        "postal_code": "string or null"
    }},
    "education": [
        {{
            "degree": "string",
            "field_of_study": "string",
            "school_institution": "string",
            "grade_cgpa_cpi_division": "string or null",
            "percentage": "string or null",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY",
            "currently_pursuing": "boolean",
            "activities_societies": "string or null",
            "degree_type": "string",
            "education_type": "string",
            "institution_type": "string",
            "institution_rating": "string or null"
        }}
    ],
    "experience_summary": {{
        "total_experience_years": "number",
        "highest_qualification_held": "string",
        "current_job_title": "string",
        "current_company_name": "string",
        "current_client_name": "string or null",
        "current_client_industry": "string or null",
        "current_employment_type": "string",
        "current_salary": "string or null",
        "expected_salary": "string or null",
        "linkedin_url": "string or null",
        "additional_info": "string or null",
        "current_timezone": "string or null",
        "current_location": "string or null",
        "resume_title": "string or null",
        "resume_synopsis": "string or null"
    }},
    "work_experience": [
        {{
            "title_role": "string",
            "engagement_type": "string",
            "employment_type": "string",
            "company_name": "string",
            "client_name": "string or null",
            "client_location": "string or null",
            "client_rating": "string or null",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY",
            "currently_working": "boolean",
            "industry_domain": "string or null",
            "industry_sub_domain": "string or null",
            "description": "string or null"
        }}
    ],
    "skills": [
        {{
            "skill_name": "string",
            "skill_main_type": "string or null",
            "skill_sub_type": "string or null",
            "skill_version": "string or null",
            "proficiency_level": "string or null",
            "experience_years": "number or null",
            "last_version_used": "string or null",
            "currently_using": "boolean or null",
            "skill_last_used": "MM/YYYY or null",
            "certified": "boolean or null"
        }}
    ],
    "certifications": [
        {{
            "certification_name": "string",
            "skill_tested": "string or null",
            "skill_type_tested": "string or null",
            "issuing_organisation": "string",
            "issue_date": "MM/YYYY",
            "url_certification_id": "string or null",
            "grade_percentage": "string or null",
            "issuer_rating": "string or null"
        }}
    ],
    "projects": [
        {{
            "project_name": "string",
            "company_name": "string or null",
            "client_name": "string or null",
            "client_location": "string or null",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY",
            "project_url": "string or null",
            "industry_domain": "string or null",
            "role": "string or null",
            "project_type": "string",
            "project_status": "string",
            "project_site": "string or null"
        }}
    ],
    "awards": [
        {{
            "name": "string",
            "associated_with": "string or null",
            "issuer": "string or null",
            "date_of_issue": "MM/YYYY",
            "description": "string or null"
        }}
    ],
    "industry_domains": [
        {{
            "domain": "string",
            "years_experience": "number",
            "currently_involved": "boolean",
            "sub_domain": "string or null",
            "last_involved": "MM/YYYY or null",
            "description": "string or null"
        }}
    ],
    "additional_info": {{
        "candidate_source": "string or null",
        "source_name": "string or null",
        "skype_id": "string or null",
        "twitter_handle": "string or null",
        "email_opt_out": "boolean",
        "candidate_status": "string or null",
        "candidate_owner": "string or null"
    }},
    "availability": {{
        "notice_period": "string or null",
        "availability_from_date": "MM/YYYY or null",
        "availability_to_date": "MM/YYYY or null",
        "part_time_availability_from": "string or null",
        "part_time_availability_to": "string or null",
        "part_time_availability_timezone": "string or null",
        "timezone": "string or null",
        "availability_types": ["array of strings"]
    }},
    "preferences": {{
        "onsite_offsite_preference": "string or null",
        "work_mode_preference": "string or null",
        "willing_relocate_india": "boolean",
        "willing_relocate_overseas": "boolean",
        "right_to_work_document": "string or null",
        "location_preferences": [
            {{
                "location_type": "string",
                "location_name": "string"
            }}
        ],
        "right_to_work_countries": ["array of strings"]
    }},
    "public_profiles": [
        {{
            "platform_name": "string",
            "url": "string",
            "platform_type": "string or null",
            "detail_type": "string or null"
        }}
    ],
    "cross_references": [
        {{
            "submitted_by": "string or null",
            "client": "string or null",
            "job_opening": "string or null",
            "status": "string or null"
        }}
    ],
    "attachments": [
        {{
            "type": "string",
            "url": "string"
        }}
    ]
}}

Resume Text:
{text}
"""

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You extract structured data from resumes with high accuracy, strictly following the provided schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        parsed_data = json.loads(response.choices[0].message.content)
        return validate_and_clean_parsed_data(parsed_data)
        
    except Exception as e:
        print(f"Llama parsing failed: {str(e)}")
        return parse_resume_with_gpt(text)

def parse_resume_with_gpt(text: str) -> dict:
    """Fallback resume parser using OpenAI GPT-4."""
    try:
        prompt = f"""
        You are an expert resume parser. Extract all possible details from the resume text 
        and structure them according to the comprehensive 14-table schema provided earlier.
        
        Return the data in a JSON format that strictly matches the schema.
        
        Resume Text:
        {text}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured data from resumes, strictly following the provided schema."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        parsed_data = json.loads(response.choices[0].message.content)
        return validate_and_clean_parsed_data(parsed_data)
    except Exception as e:
        raise Exception(f"Error parsing resume: {str(e)}")

def validate_and_clean_parsed_data(parsed_data: dict) -> dict:
    """Validate and clean the parsed data to ensure it matches the schema."""
    # Ensure all top-level sections exist
    for section in ['basic_info', 'address', 'education', 'experience_summary', 
                   'work_experience', 'skills', 'certifications', 'projects', 
                   'awards', 'industry_domains', 'additional_info', 'availability',
                   'preferences', 'public_profiles', 'cross_references', 'attachments']:
        if section not in parsed_data:
            if section in ['basic_info', 'address', 'experience_summary', 'additional_info', 
                          'availability', 'preferences']:
                parsed_data[section] = {}
            else:
                parsed_data[section] = []
    
    # Clean arrays to ensure they contain dictionaries
    for array_section in ['education', 'work_experience', 'skills', 'certifications', 
                         'projects', 'awards', 'industry_domains', 'public_profiles',
                         'cross_references', 'attachments']:
        if not isinstance(parsed_data[array_section], list):
            parsed_data[array_section] = []
        else:
            parsed_data[array_section] = [
                {k: v for k, v in item.items() if v is not None and v != ""}
                for item in parsed_data[array_section]
            ]
    
    # Clean nested dictionaries
    for dict_section in ['basic_info', 'address', 'experience_summary', 'additional_info',
                         'availability', 'preferences']:
        if not isinstance(parsed_data[dict_section], dict):
            parsed_data[dict_section] = {}
        else:
            parsed_data[dict_section] = {
                k: v for k, v in parsed_data[dict_section].items() 
                if v is not None and v != "" and v != [] and v != {}
            }
    
    return parsed_data

def get_embedding(text: str) -> np.ndarray:
    """Get text embedding using OpenAI's embedding model."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002", 
        input=text
    )
    return np.array(response.data[0].embedding)

def calculate_similarity_score(resume_data: dict, jd_text: str) -> float:
    """Calculate cosine similarity between resume and job description."""
    resume_text_parts = []
    
    # Basic info
    if 'basic_info' in resume_data:
        basic = resume_data['basic_info']
        resume_text_parts.append(f"Name: {basic.get('first_name', '')} {basic.get('last_name', '')}")
        resume_text_parts.append(f"Contact: {basic.get('email', '')} | {basic.get('mobile', '')}")
    
    # Experience summary
    if 'experience_summary' in resume_data:
        exp = resume_data['experience_summary']
        resume_text_parts.append(f"Summary: {exp.get('resume_synopsis', '')}")
        resume_text_parts.append(f"Current Role: {exp.get('current_job_title', '')} at {exp.get('current_company_name', '')}")
        resume_text_parts.append(f"Experience: {exp.get('total_experience_years', '')} years")
    
    # Work experience
    if 'work_experience' in resume_data:
        for job in resume_data['work_experience']:
            resume_text_parts.append(
                f"Role: {job.get('title_role', '')} at {job.get('company_name', '')} "
                f"({job.get('start_date', '')} to {job.get('end_date', '')}): "
                f"{job.get('description', '')}"
            )
    
    # Skills
    if 'skills' in resume_data:
        skills_text = ", ".join(
            f"{skill.get('skill_name', '')} ({skill.get('proficiency_level', '')})"
            for skill in resume_data['skills']
        )
        resume_text_parts.append(f"Skills: {skills_text}")
    
    # Education
    if 'education' in resume_data:
        for edu in resume_data['education']:
            resume_text_parts.append(
                f"Education: {edu.get('degree', '')} from {edu.get('school_institution', '')}"
            )
    
    resume_text = " ".join(resume_text_parts)
    
    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)
    return round(float(cosine_similarity([resume_emb], [jd_emb])[0][0]) * 100, 2)

def store_candidate_data(parsed_data: dict) -> str:
    """Store parsed resume data in all 14 tables and return candidate UUID."""
    conn = None
    candidate_id = None
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate candidate UUID
        candidate_id = str(uuid.uuid4())
        
        # 1. Insert basic candidate info
        basic_info = parsed_data.get('basic_info', {})
        cur.execute(
            """
            INSERT INTO candidates (
                candidate_id, photo, salutation, first_name, middle_name, last_name, 
                email, secondary_email, mobile, secondary_mobile
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                candidate_id,
                basic_info.get('photo'),
                basic_info.get('salutation'),
                basic_info.get('first_name'),
                basic_info.get('middle_name'),
                basic_info.get('last_name'),
                basic_info.get('email'),
                basic_info.get('secondary_email'),
                basic_info.get('mobile'),
                basic_info.get('secondary_mobile')
            )
        )
        
        # 2. Insert address
        address = parsed_data.get('address', {})
        cur.execute(
            """
            INSERT INTO addresses (
                address_id, candidate_id, address_line_1, address_line_2, 
                city, state, country, postal_code
            ) VALUES (uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                candidate_id,
                address.get('address_line_1'),
                address.get('address_line_2'),
                address.get('city'),
                address.get('state'),
                address.get('country'),
                address.get('postal_code')
            )
        )
        
        # 3. Insert education
        for edu in parsed_data.get('education', []):
            cur.execute(
                """
                INSERT INTO education (
                    education_id, candidate_id, degree, field_of_study, school_institution,
                    grade_cgpa_cpi_division, percentage, start_date, end_date, 
                    currently_pursuing, activities_societies, degree_type, 
                    education_type, institution_type, institution_rating
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    edu.get('degree'),
                    edu.get('field_of_study'),
                    edu.get('school_institution'),
                    edu.get('grade_cgpa_cpi_division'),
                    edu.get('percentage'),
                    edu.get('start_date'),
                    edu.get('end_date'),
                    edu.get('currently_pursuing', False),
                    edu.get('activities_societies'),
                    edu.get('degree_type'),
                    edu.get('education_type'),
                    edu.get('institution_type'),
                    edu.get('institution_rating')
                )
            )
        
        # 4. Insert experience summary
        exp_summary = parsed_data.get('experience_summary', {})
        cur.execute(
            """
            INSERT INTO experience_summary (
                summary_id, candidate_id, total_experience_years, highest_qualification_held,
                current_job_title, current_company_name, current_client_name,
                current_client_industry, current_employment_type, current_salary,
                expected_salary, linkedin_url, additional_info, current_timezone,
                current_location, resume_title, resume_synopsis
            ) VALUES (
                uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                candidate_id,
                exp_summary.get('total_experience_years'),
                exp_summary.get('highest_qualification_held'),
                exp_summary.get('current_job_title'),
                exp_summary.get('current_company_name'),
                exp_summary.get('current_client_name'),
                exp_summary.get('current_client_industry'),
                exp_summary.get('current_employment_type'),
                exp_summary.get('current_salary'),
                exp_summary.get('expected_salary'),
                exp_summary.get('linkedin_url'),
                exp_summary.get('additional_info'),
                exp_summary.get('current_timezone'),
                exp_summary.get('current_location'),
                exp_summary.get('resume_title'),
                exp_summary.get('resume_synopsis')
            )
        )
        
        # 5. Insert work experience
        for exp in parsed_data.get('work_experience', []):
            cur.execute(
                """
                INSERT INTO work_experience (
                    experience_id, candidate_id, title_role, engagement_type, 
                    employment_type, company_name, client_name, client_location,
                    client_rating, start_date, end_date, currently_working,
                    industry_domain, industry_sub_domain, description
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    exp.get('title_role'),
                    exp.get('engagement_type'),
                    exp.get('employment_type'),
                    exp.get('company_name'),
                    exp.get('client_name'),
                    exp.get('client_location'),
                    exp.get('client_rating'),
                    exp.get('start_date'),
                    exp.get('end_date'),
                    exp.get('currently_working', False),
                    exp.get('industry_domain'),
                    exp.get('industry_sub_domain'),
                    exp.get('description')
                )
            )
        
        # 6. Insert skills
        for skill in parsed_data.get('skills', []):
            cur.execute(
                """
                INSERT INTO skills (
                    skill_id, candidate_id, skill_name, skill_main_type, skill_sub_type,
                    skill_version, proficiency_level, experience_years, last_version_used,
                    currently_using, skill_last_used, certified
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    skill.get('skill_name'),
                    skill.get('skill_main_type'),
                    skill.get('skill_sub_type'),
                    skill.get('skill_version'),
                    skill.get('proficiency_level'),
                    skill.get('experience_years'),
                    skill.get('last_version_used'),
                    skill.get('currently_using', False),
                    skill.get('skill_last_used'),
                    skill.get('certified', False)
                )
            )
        
        # 7. Insert certifications
        for cert in parsed_data.get('certifications', []):
            cur.execute(
                """
                INSERT INTO certifications (
                    certification_id, candidate_id, certification_name, skill_tested,
                    skill_type_tested, issuing_organisation, issue_date,
                    url_certification_id, grade_percentage, issuer_rating
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    cert.get('certification_name'),
                    cert.get('skill_tested'),
                    cert.get('skill_type_tested'),
                    cert.get('issuing_organisation'),
                    cert.get('issue_date'),
                    cert.get('url_certification_id'),
                    cert.get('grade_percentage'),
                    cert.get('issuer_rating')
                )
            )
        
        # 8. Insert projects
        for project in parsed_data.get('projects', []):
            cur.execute(
                """
                INSERT INTO projects (
                    project_id, candidate_id, project_name, company_name, client_name,
                    client_location, start_date, end_date, project_url, industry_domain,
                    role, project_type, project_status, project_site
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    project.get('project_name'),
                    project.get('company_name'),
                    project.get('client_name'),
                    project.get('client_location'),
                    project.get('start_date'),
                    project.get('end_date'),
                    project.get('project_url'),
                    project.get('industry_domain'),
                    project.get('role'),
                    project.get('project_type'),
                    project.get('project_status'),
                    project.get('project_site')
                )
            )
        
        # 9. Insert awards
        for award in parsed_data.get('awards', []):
            cur.execute(
                """
                INSERT INTO awards (
                    award_id, candidate_id, name, associated_with, issuer, date_of_issue, description
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    award.get('name'),
                    award.get('associated_with'),
                    award.get('issuer'),
                    award.get('date_of_issue'),
                    award.get('description')
                )
            )
        
        # 10. Insert industry domains
        for domain in parsed_data.get('industry_domains', []):
            cur.execute(
                """
                INSERT INTO industry_domains (
                    domain_id, candidate_id, domain, years_experience, currently_involved,
                    sub_domain, last_involved, description
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    domain.get('domain'),
                    domain.get('years_experience'),
                    domain.get('currently_involved', False),
                    domain.get('sub_domain'),
                    domain.get('last_involved'),
                    domain.get('description')
                )
            )
        
        # 11. Insert additional info
        additional_info = parsed_data.get('additional_info', {})
        cur.execute(
            """
            INSERT INTO additional_info (
                info_id, candidate_id, candidate_source, source_name, skype_id,
                twitter_handle, email_opt_out, candidate_status, candidate_owner
            ) VALUES (
                uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                candidate_id,
                additional_info.get('candidate_source'),
                additional_info.get('source_name'),
                additional_info.get('skype_id'),
                additional_info.get('twitter_handle'),
                additional_info.get('email_opt_out', False),
                additional_info.get('candidate_status'),
                additional_info.get('candidate_owner')
            )
        )
        
        # 12. Insert availability
        availability = parsed_data.get('availability', {})
        cur.execute(
            """
            INSERT INTO availability (
                availability_id, candidate_id, notice_period, availability_from_date,
                availability_to_date, part_time_availability_from, part_time_availability_to,
                part_time_availability_timezone, timezone
            ) VALUES (
                uuid_generate_v4(), %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING availability_id
            """,
            (
                candidate_id,
                availability.get('notice_period'),
                availability.get('availability_from_date'),
                availability.get('availability_to_date'),
                availability.get('part_time_availability_from'),
                availability.get('part_time_availability_to'),
                availability.get('part_time_availability_timezone'),
                availability.get('timezone')
            )
        )
        availability_id = cur.fetchone()[0]
        
        # 12a. Insert availability types
        for avail_type in availability.get('availability_types', []):
            cur.execute(
                """
                INSERT INTO availability_types (
                    type_id, availability_id, availability_type
                ) VALUES (
                    uuid_generate_v4(), %s, %s
                )
                """,
                (availability_id, avail_type)
            )
        
        # 13. Insert preferences
        preferences = parsed_data.get('preferences', {})
        cur.execute(
            """
            INSERT INTO preferences (
                preference_id, candidate_id, onsite_offsite_preference, work_mode_preference,
                willing_relocate_india, willing_relocate_overseas, right_to_work_document
            ) VALUES (
                uuid_generate_v4(), %s, %s, %s, %s, %s, %s
            )
            RETURNING preference_id
            """,
            (
                candidate_id,
                preferences.get('onsite_offsite_preference'),
                preferences.get('work_mode_preference'),
                preferences.get('willing_relocate_india', False),
                preferences.get('willing_relocate_overseas', False),
                preferences.get('right_to_work_document')
            )
        )
        preference_id = cur.fetchone()[0]
        
        # 13a. Insert location preferences
        for loc_pref in preferences.get('location_preferences', []):
            cur.execute(
                """
                INSERT INTO location_preferences (
                    location_id, preference_id, location_type, location_name
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s
                )
                """,
                (
                    preference_id,
                    loc_pref.get('location_type'),
                    loc_pref.get('location_name')
                )
            )
        
        # 13b. Insert right to work countries
        for country in preferences.get('right_to_work_countries', []):
            cur.execute(
                """
                INSERT INTO right_to_work_countries (
                    country_id, preference_id, country_name
                ) VALUES (
                    uuid_generate_v4(), %s, %s
                )
                """,
                (preference_id, country)
            )
        
        # 14. Insert public profiles
        for profile in parsed_data.get('public_profiles', []):
            cur.execute(
                """
                INSERT INTO public_profiles (
                    profile_id, candidate_id, platform_name, url, platform_type, detail_type
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    profile.get('platform_name'),
                    profile.get('url'),
                    profile.get('platform_type'),
                    profile.get('detail_type')
                )
            )
        
        # 15. Insert cross references
        for ref in parsed_data.get('cross_references', []):
            cur.execute(
                """
                INSERT INTO cross_references (
                    reference_id, candidate_id, submitted_by, client, job_opening, status
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    ref.get('submitted_by'),
                    ref.get('client'),
                    ref.get('job_opening'),
                    ref.get('status')
                )
            )
        
        # 16. Insert attachments
        for attachment in parsed_data.get('attachments', []):
            cur.execute(
                """
                INSERT INTO attachments (
                    attachment_id, candidate_id, type, url
                ) VALUES (
                    uuid_generate_v4(), %s, %s, %s
                )
                """,
                (
                    candidate_id,
                    attachment.get('type'),
                    attachment.get('url')
                )
            )
        
        conn.commit()
        return candidate_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Error storing candidate data: {str(e)}")
    finally:
        if conn:
            cur.close()
            conn.close()

def process_single_resume(file, jd_text=None):
    """Process a single resume file with optional JD comparison."""
    try:
        file_path = Path(UPLOAD_FOLDER) / file.filename
        file.save(file_path)
        resume_text = extract_text(file_path)
        
        # Parse resume
        parsed_resume = parse_resume_with_llama(resume_text)
        
        # Store in database
        candidate_id = store_candidate_data(parsed_resume)
        
        similarity_score = None
        if jd_text:
            similarity_score = calculate_similarity_score(parsed_resume, jd_text)
        
        # Clean up the file after processing
        file_path.unlink(missing_ok=True)
        
        return {
            "filename": file.filename,
            "candidate_id": candidate_id,
            "similarity_score": similarity_score,
            "error": None
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "candidate_id": None,
            "similarity_score": None,
            "error": str(e)
        }

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    """Main endpoint for resume processing."""
    if request.method == "POST":
        try:
            files = request.files.getlist("resume")
            jd_text = request.form.get("jd_text", "").strip()
            
            if not files:
                return render_template("index.html", error="No resume files uploaded")
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_single_resume, file, jd_text) for file in files]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            # Sort by similarity score if JD provided
            if jd_text:
                results.sort(key=lambda x: x["similarity_score"] or 0, reverse=True)
            
            return render_template("index.html", 
                                results=results,
                                jd_text=jd_text if jd_text else None)
        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html")

@app.route("/api/parse", methods=["POST"])
def api_parse_resume():
    """API endpoint for resume parsing."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        jd_text = request.form.get("jd_text", "").strip()
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        result = process_single_resume(file, jd_text if jd_text else None)
        
        if result["error"]:
            return jsonify({"error": result["error"]}), 500
        
        return jsonify({
            "filename": result["filename"],
            "candidate_id": result["candidate_id"],
            "similarity_score": result["similarity_score"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/candidate/<string:candidate_id>", methods=["GET"])
def get_candidate(candidate_id):
    """API endpoint to get candidate data from database."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get basic candidate info
        cur.execute("SELECT * FROM candidates WHERE candidate_id = %s", (candidate_id,))
        candidate = dict(cur.fetchone())
        
        # Get all related data from all 14 tables
        tables = [
            ('addresses', 'address'),
            ('education', 'education'),
            ('experience_summary', 'experience_summary'),
            ('work_experience', 'work_experience'),
            ('skills', 'skills'),
            ('certifications', 'certifications'),
            ('projects', 'projects'),
            ('awards', 'awards'),
            ('industry_domains', 'industry_domains'),
            ('additional_info', 'additional_info'),
            ('availability', 'availability'),
            ('preferences', 'preferences'),
            ('public_profiles', 'public_profiles'),
            ('cross_references', 'cross_references'),
            ('attachments', 'attachments')
        ]
        
        for table, key in tables:
            cur.execute(f"SELECT * FROM {table} WHERE candidate_id = %s", (candidate_id,))
            candidate[key] = [dict(row) for row in cur.fetchall()]
        
        # Get nested data for availability types
        if 'availability' in candidate and len(candidate['availability']) > 0:
            avail_id = candidate['availability'][0]['availability_id']
            cur.execute("SELECT availability_type FROM availability_types WHERE availability_id = %s", (avail_id,))
            candidate['availability'][0]['availability_types'] = [row[0] for row in cur.fetchall()]
        
        # Get nested data for location preferences
        if 'preferences' in candidate and len(candidate['preferences']) > 0:
            pref_id = candidate['preferences'][0]['preference_id']
            cur.execute("SELECT location_type, location_name FROM location_preferences WHERE preference_id = %s", (pref_id,))
            candidate['preferences'][0]['location_preferences'] = [dict(row) for row in cur.fetchall()]
            
            cur.execute("SELECT country_name FROM right_to_work_countries WHERE preference_id = %s", (pref_id,))
            candidate['preferences'][0]['right_to_work_countries'] = [row[0] for row in cur.fetchall()]
        
        return jsonify(candidate)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)