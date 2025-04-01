import os
import psycopg2
from dotenv import load_dotenv, find_dotenv

# Load environment variables
env_path = find_dotenv('D:/lifeline/api.env')
load_dotenv(env_path)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )


    
def create_database_schema():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        print("Creating database schema...")
        
        # Enable UUID extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        
        # 1. Candidates (Basic Info)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            photo TEXT,
            salutation TEXT,
            first_name TEXT NOT NULL,
            middle_name TEXT,
            last_name TEXT NOT NULL,
            email TEXT ,
            secondary_email TEXT,
            mobile TEXT NOT NULL,
            secondary_mobile TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT email_unique UNIQUE (email)
        )
        """)
        
        # 2. Addresses
        cur.execute("""
        CREATE TABLE IF NOT EXISTS addresses (
            address_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            address_line_1 TEXT,
            address_line_2 TEXT,
            city TEXT,
            state TEXT,
            country TEXT,
            postal_code TEXT
        )
        """)
        
        # 3. Education
        cur.execute("""
        CREATE TABLE IF NOT EXISTS education (
            education_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            degree TEXT,
            field_of_study TEXT,
            school_institution TEXT,
            grade_cgpa_cpi_division TEXT,
            percentage TEXT,
            start_date TEXT,
            end_date TEXT,
            currently_pursuing BOOLEAN DEFAULT FALSE,
            activities_societies TEXT,
            degree_type TEXT,
            education_type TEXT,
            institution_type TEXT,
            institution_rating TEXT
        )
        """)
        
        # 4. Experience Summary
        cur.execute("""
        CREATE TABLE IF NOT EXISTS experience_summary (
            summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            total_experience_years FLOAT,
            highest_qualification_held TEXT,
            current_job_title TEXT,
            current_company_name TEXT,
            current_client_name TEXT,
            current_client_industry TEXT,
            current_employment_type TEXT,
            current_salary TEXT,
            expected_salary TEXT,
            linkedin_url TEXT,
            additional_info TEXT,
            current_timezone TEXT,
            current_location TEXT,
            resume_title TEXT,
            resume_synopsis TEXT
        )
        """)
        
        # 5. Work Experience
        cur.execute("""
        CREATE TABLE IF NOT EXISTS work_experience (
            experience_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            title_role TEXT,
            engagement_type TEXT,
            employment_type TEXT,
            company_name TEXT,
            client_name TEXT,
            client_location TEXT,
            client_rating TEXT,
            start_date TEXT,
            end_date TEXT,
            currently_working BOOLEAN DEFAULT FALSE,
            industry_domain TEXT,
            industry_sub_domain TEXT,
            description TEXT
        )
        """)
        
        # 6. Skills
        cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            skill_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            skill_name TEXT NOT NULL,
            skill_main_type TEXT,
            skill_sub_type TEXT,
            skill_version TEXT,
            proficiency_level TEXT,
            experience_years FLOAT,
            last_version_used TEXT,
            currently_using BOOLEAN DEFAULT FALSE,
            skill_last_used TEXT,
            certified BOOLEAN DEFAULT FALSE
        )
        """)
        
        # 7. Certifications
        cur.execute("""
        CREATE TABLE IF NOT EXISTS certifications (
            certification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            certification_name TEXT NOT NULL,
            skill_tested TEXT,
            skill_type_tested TEXT,
            issuing_organisation TEXT,
            issue_date TEXT,
            url_certification_id TEXT,
            grade_percentage TEXT,
            issuer_rating TEXT
        )
        """)
        
        # 8. Projects
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            project_name TEXT,
            company_name TEXT,
            client_name TEXT,
            client_location TEXT,
            start_date TEXT,
            end_date TEXT,
            project_url TEXT,
            industry_domain TEXT,
            role TEXT,
            project_type TEXT,
            project_status TEXT,
            project_site TEXT
        )
        """)
        
        # 9. Awards
        cur.execute("""
        CREATE TABLE IF NOT EXISTS awards (
            award_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            associated_with TEXT,
            issuer TEXT,
            date_of_issue TEXT,
            description TEXT
        )
        """)
        
        # 10. Industry Domains
        cur.execute("""
        CREATE TABLE IF NOT EXISTS industry_domains (
            domain_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            domain TEXT,
            years_experience FLOAT,
            currently_involved BOOLEAN DEFAULT FALSE,
            sub_domain TEXT,
            last_involved TEXT,
            description TEXT
        )
        """)
        
        # 11. Additional Info
        cur.execute("""
        CREATE TABLE IF NOT EXISTS additional_info (
            info_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            candidate_source TEXT,
            source_name TEXT,
            skype_id TEXT,
            twitter_handle TEXT,
            email_opt_out BOOLEAN DEFAULT FALSE,
            candidate_status TEXT,
            candidate_owner TEXT
        )
        """)
        
        # 12. Availability
        cur.execute("""
        CREATE TABLE IF NOT EXISTS availability (
            availability_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            notice_period TEXT,
            availability_from_date TEXT,
            availability_to_date TEXT,
            part_time_availability_from TEXT,
            part_time_availability_to TEXT,
            part_time_availability_timezone TEXT,
            timezone TEXT
        )
        """)
        
        # 12a. Availability Types
        cur.execute("""
        CREATE TABLE IF NOT EXISTS availability_types (
            type_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            availability_id UUID REFERENCES availability(availability_id) ON DELETE CASCADE,
            availability_type TEXT NOT NULL
        )
        """)
        
        # 13. Preferences
        cur.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            preference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            onsite_offsite_preference TEXT,
            work_mode_preference TEXT,
            willing_relocate_india BOOLEAN DEFAULT FALSE,
            willing_relocate_overseas BOOLEAN DEFAULT FALSE,
            right_to_work_document TEXT
        )
        """)
        
        # 13a. Location Preferences
        cur.execute("""
        CREATE TABLE IF NOT EXISTS location_preferences (
            location_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            preference_id UUID REFERENCES preferences(preference_id) ON DELETE CASCADE,
            location_type TEXT,
            location_name TEXT NOT NULL
        )
        """)
        
        # 13b. Right to Work Countries
        cur.execute("""
        CREATE TABLE IF NOT EXISTS right_to_work_countries (
            country_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            preference_id UUID REFERENCES preferences(preference_id) ON DELETE CASCADE,
            country_name TEXT NOT NULL
        )
        """)
        
        # 14. Public Profiles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS public_profiles (
            profile_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            platform_name TEXT NOT NULL,
            url TEXT NOT NULL,
            platform_type TEXT,
            detail_type TEXT
        )
        """)
        
        # 15. Cross References
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cross_references (
            reference_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            submitted_by TEXT,
            client TEXT,
            job_opening TEXT,
            status TEXT
        )
        """)
        
        # 16. Attachments
        cur.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            attachment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            candidate_id UUID REFERENCES candidates(candidate_id) ON DELETE CASCADE,
            type TEXT,
            url TEXT NOT NULL
        )
        """)
        
        conn.commit()
        print("Database schema created successfully")
    except Exception as e:
        print(f"Error creating database schema: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            cur.close()
            conn.close()

# Create schema on startup
create_database_schema() 