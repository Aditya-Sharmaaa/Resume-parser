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

# Load environment variables
env_path = find_dotenv('D:/lifeline/api.env')
load_dotenv(env_path)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_PATH = Path(r"C:\Users\acer\Downloads\data")
UPLOAD_PATH.mkdir(exist_ok=True)

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

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
    """Parse resume text using Groq's Llama 70B for faster processing."""
    try:
        prompt = f"""
        You are an expert resume parser. Extract the following details in JSON format:
        {{
            "full_name": "string",
            "email": "string",
            "github": "url or null",
            "linkedin": "url or null",
            "certification": "string or null",
            "location": "string or null",
            "projects": "string or null",
            "phone": "string or null",
            "experience_summary": "string or null",
            "employment": [
                {{
                    "company": "string",
                    "title": "string",
                    "start_date": "string",
                    "end_date": "string",
                    "description": "string"
                }}
            ],
            "technical_skills": ["array", "of", "strings"],
            "soft_skills": ["array", "of", "strings"]
        }}
        
        Resume Text:
        {text}
        """

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You extract structured data from resumes with high accuracy."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # Fallback to GPT-4 if Llama fails
        print(f"Llama parsing failed, falling back to GPT-4: {str(e)}")
        return parse_resume_with_gpt(text)

def parse_resume_with_gpt(text: str) -> dict:
    """Fallback resume parser using OpenAI GPT-4."""
    try:
        prompt = f"""
        You are an expert resume parser. Extract the following details:
        - full_name
        - email
        - github (url or null)
        - linkedin (url or null)
        - certification
        - location
        - projects
        - phone
        - experience_summary
        - employment: [{{ company, title, start_date, end_date, description }}]
        - technical_skills (array)
        - soft_skills (array)
        
        Resume Text:
        {text}

        Return the extracted information in JSON format.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract structured data from resumes."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise Exception(f"Error parsing resume: {str(e)}")

def get_embedding(text: str) -> np.ndarray:
    """Get text embedding using OpenAI's embedding model."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002", 
        input=text
    )
    return np.array(response.data[0].embedding)

def calculate_similarity_score(resume_text: str, jd_text: str) -> float:
    """Calculate cosine similarity between resume and job description."""
    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(jd_text)
    return round(float(cosine_similarity([resume_emb], [jd_emb])[0][0]) * 100, 2)

def process_single_resume(file, jd_text=None):
    """Process a single resume file with optional JD comparison."""
    try:
        file_path = UPLOAD_PATH / file.filename
        file.save(file_path)
        resume_text = extract_text(file_path)
        
        # Use Llama 70B for faster parsing
        parsed_resume = parse_resume_with_llama(resume_text)
        
        similarity_score = None
        if jd_text:
            similarity_score = calculate_similarity_score(resume_text, jd_text)
        
        # Clean up the file after processing
        file_path.unlink(missing_ok=True)
        
        return {
            "filename": file.filename,
            "parsed_data": parsed_resume,
            "similarity_score": similarity_score,
            "error": None
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "parsed_data": None,
            "similarity_score": None,
            "error": str(e)
        }

def process_resumes_batch(files, jd_text=None):
    """Process multiple resumes in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for file in files:
            futures.append(executor.submit(process_single_resume, file, jd_text))
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Sort results by similarity score if JD text was provided
    if jd_text:
        results.sort(key=lambda x: x["similarity_score"] or 0, reverse=True)
    
    return results

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    """Main endpoint for resume processing."""
    if request.method == "POST":
        try:
            files = request.files.getlist("resume")
            jd_text = request.form.get("jd_text", "").strip()
            
            if not files:
                return render_template("index.html", error="No resume files uploaded")
            
            results = process_resumes_batch(files, jd_text if jd_text else None)
            
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
            "parsed_data": result["parsed_data"],
            "similarity_score": result["similarity_score"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)