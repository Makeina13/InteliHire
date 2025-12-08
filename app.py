from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
import PyPDF2

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_JOB_DESCRIPTION_LENGTH = 10000

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_cv_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    return ""


def analyze_cv_with_ai(cv_text, job_description):
    prompt = f"""You are an expert CV writer and career advisor. Your job is to REWRITE and IMPROVE the user's CV to match the job description.

CV Content:
{cv_text}

Job Description:
{job_description}

STEP 1 - REALISTIC FIT CHECK (CRITICAL):
First, determine if this CV can realistically be adapted for this role.

UNSUITABLE MATCHES (Score below 30, DO NOT rewrite):
- Art/Design CV applying for Software Engineering
- Healthcare CV applying for Finance/Banking
- Teaching CV applying for Data Science (without relevant skills)
- Any CV where the core experience is completely unrelated

If unsuitable, respond ONLY with:
"MATCH STATUS: NOT SUITABLE

I cannot rewrite this CV for this role because [explain why in 2-3 sentences].

Your background is primarily in [their field]. This role requires [key job requirements].

RECOMMENDED ALTERNATIVE PATHS:
1. [Alternative career path that fits their background]
2. [Skills they could learn to transition - be specific]
3. [Related roles that might be a better fit]

If you want to transition to this field in the future, consider:
- [Specific course or certification]
- [Type of project they could build]
- [Entry-level role to start with]"

STEP 2 - IF SUITABLE, REWRITE THE CV:
If the CV CAN be adapted (even if it needs significant work), provide the following:

MATCH STATUS: SUITABLE
Match Score: [40-100]/100

IMPROVED PROFESSIONAL SUMMARY
(Copy this to replace your current summary)

[Write a compelling 3-4 sentence professional summary tailored specifically to this job. Include years of experience, key skills, and what value they bring.]

IMPROVED EXPERIENCE SECTION
(Copy these improved bullet points for each role)

[Job Title] at [Company]
- [Rewritten bullet using STAR format with specific metrics]
- [Rewritten bullet with quantifiable achievement]
- [Rewritten bullet highlighting skill relevant to target job]

[Repeat for other relevant positions]

SKILLS SECTION
(Update your skills section to include these)

Technical Skills: [List relevant technical skills they have]
Soft Skills: [List relevant soft skills demonstrated in their CV]
Add These If You Have Them: [Skills from job description they should add if applicable]

SECTIONS TO ADD

[Suggest any sections they should add - Projects, Certifications, Volunteer Work, etc. with specific examples of what to include]

SECTIONS TO REMOVE OR REDUCE

[List any sections that are irrelevant or taking up valuable space]

QUICK SUMMARY OF CHANGES MADE

1. [Key change #1 and why it improves the CV]
2. [Key change #2 and why it improves the CV]
3. [Key change #3 and why it improves the CV]

Remember: All improved content should be COPY-PASTE READY. The user should be able to directly use these sections in their CV."""
    response = model.generate_content(prompt)
    return response.text


@app.route('/')
def index():
    return render_template('landingpage.html')


@app.route('/upload')
def upload_cv():
    return render_template('Cvupload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'cv' not in request.files:
        return jsonify({'error': 'No CV file uploaded'}), 400

    file = request.files['cv']
    job_description = request.form.get('job_description', '')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PDF or DOCX'}), 400

    if not job_description:
        return jsonify({'error': 'Job description is required'}), 400

    if len(job_description) > MAX_JOB_DESCRIPTION_LENGTH:
        return jsonify({'error': 'Job description too long. Maximum 10,000 characters.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        cv_text = extract_cv_text(filepath)

        if not cv_text or len(cv_text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from CV. Please check the file.'}), 400

        analysis = analyze_cv_with_ai(cv_text, job_description)

        os.remove(filepath)

        return jsonify({
            'success': True,
            'analysis': analysis
        })

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500


@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=False) 