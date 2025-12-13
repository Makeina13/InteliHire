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
    prompt = f"""Rewrite this CV to match the job. You must write like an actual human, not AI.

STRICT RULES TO AVOID AI DETECTION:

1. Use contractions: "I'm" not "I am", "don't" not "do not", "I've" not "I have"
2. Start sentences differently - not everything with "I"
3. Use casual connecting words: "Also", "Plus", "On top of that", "Honestly"
4. Include filler phrases humans use: "pretty much", "a lot of", "kind of", "basically"
5. Make small imperfections - don't be too polished
6. Use shorter sentences mixed with longer ones
7. Be specific and personal, not generic
8. Avoid these AI words completely: passionate, driven, dedicated, motivated, thrive, leverage, spearhead, utilize, facilitate, innovative, dynamic, synergy, proactive, endeavor, comprehensive, enhance, foster, implement, streamline, optimize
9. Sound like a real person talking about their job, not a LinkedIn post
10. Use "got" instead of "obtained", "helped" instead of "assisted", "worked on" instead of "contributed to"

BAD EXAMPLE (AI-sounding):
"I am a motivated A-level student with a strong interest in digital innovation and data science. My studies in Mathematics have given me a solid foundation. I'm keen to apply these skills to complex challenges and contribute to the company's digital transformation goals."

GOOD EXAMPLE (Human-sounding):
"I'm finishing my A-levels in 2026 - doing Maths and Computer Science. I've been teaching myself Python and built a few small projects, nothing fancy but I learned a lot. Honestly, I just want to get stuck in somewhere and see what working in tech is actually like."

CV Content:
{cv_text}

Job Description:
{job_description}

First check if this CV fits the role. If someone with an art background applies for software engineering, or healthcare for finance, say it's not suitable.

If NOT SUITABLE:

MATCH STATUS: NOT SUITABLE

This CV doesn't fit this role because [short reason].

Your experience is in [field]. This job needs [requirements].

Better options for you:
1. [Role that fits]
2. [Skills to learn]
3. [Related jobs]

If SUITABLE:

MATCH STATUS: SUITABLE
Score: [40-100]/100

PROFESSIONAL SUMMARY

[Write 2-3 sentences MAX. Use contractions. Sound like a real person. Be specific. No buzzwords.]

EXPERIENCE

[Job] at [Company]
- [Short, natural bullet. Use numbers. Sound human.]
- [Mix up how you start each bullet]
- [Don't be too formal]

SKILLS

[Just list them simply, no fancy descriptions]

ADD THESE SECTIONS

[What to add]

REMOVE THESE

[What to cut]

WHAT I CHANGED

1. [Change]
2. [Change]
3. [Change]"""

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