from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
import PyPDF2

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

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
    prompt = f"""You are an expert career advisor and CV analyst. Analyze the following CV against the job description provided.

CV Content:
{cv_text}

Job Description:
{job_description}

CRITICAL: First, determine if the candidate's background is realistic for this role. If the CV is from a completely different field (e.g., art background applying for software engineering, or healthcare applying for finance), you MUST be honest and state that this is not a suitable match.

Please provide:

1. REALISTIC FIT ASSESSMENT:
   - Is this CV genuinely suitable for this role?
   - If not, explain why and suggest more appropriate career paths

2. SKILLS MATCH (only if realistic fit):
   - Which required skills does the candidate have?
   - Which required skills are missing?

3. EXPERIENCE ALIGNMENT (only if realistic fit):
   - Does their experience match the role requirements?
   - What relevant experience do they have?

4. GAPS & WEAKNESSES:
   - What are the major gaps?
   - What needs improvement?

5. STRENGTHS:
   - What are the candidate's strongest points?

6. RECOMMENDATIONS:
   - If suitable: Specific improvements for the CV
   - If unsuitable: Alternative career paths that match their background

7. OVERALL SCORE (0-100):
   - Only give a score above 40 if the candidate is realistically suitable
   - If unsuitable, score should be below 30

Be honest and helpful. Don't try to force-fit an unsuitable candidate."""

    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

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
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)