from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import re
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.graphics.shapes import Drawing, Line

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# --- CONFIGURATION ---
generation_config = genai.GenerationConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    response_mime_type="application/json"
)

# --- SMART MODEL LIST GENERATOR ---
def get_prioritized_model_list():
    """
    Returns a list of ALL valid model names your account has access to,
    sorted by our preference (2.0 -> Flash -> Pro).
    """
    print("--------------------------------------------------")
    print("📋 BUILDING MODEL LIST...")
    
    try:
        # 1. Get every model you actually own
        all_my_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Strip the 'models/' prefix for cleaner usage
                name = m.name.replace('models/', '')
                all_my_models.append(name)
        
        # 2. Sort them by preference
        # We want to try 2.0 first, then Flash, then Pro.
        prioritized = []
        
        # Priority 1: Gemini 2.0
        prioritized.extend([m for m in all_my_models if 'gemini-2.0' in m])
        
        # Priority 2: Flash (Any version)
        prioritized.extend([m for m in all_my_models if 'flash' in m and m not in prioritized])
        
        # Priority 3: Pro / Standard (Any version)
        prioritized.extend([m for m in all_my_models if 'pro' in m and m not in prioritized])
        
        # Priority 4: Whatever is left
        for m in all_my_models:
            if m not in prioritized:
                prioritized.append(m)
                
        print(f"✅ Found {len(prioritized)} valid models.")
        print(f"🚀 Priority Order: {prioritized}")
        print("--------------------------------------------------")
        return prioritized

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        # Emergency Fallback if list fails
        return ['gemini-2.0-flash-exp', 'gemini-flash-latest', 'gemini-pro']

# Initialize the list ONCE when server starts
VALID_MODELS = get_prioritized_model_list()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_cv_text(file_path):
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading file: {e}")
    return text

def get_line(width=480, color="#e2e8f0"):
    d = Drawing(width, 1)
    d.add(Line(0, 0, width, 0, strokeColor=colors.HexColor(color)))
    return d

def create_pdf(data, filename):
    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
    doc = SimpleDocTemplate(filepath, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    primary_col = colors.HexColor("#0f172a")
    accent_col = colors.HexColor("#0ea5e9")
    
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=24, textColor=primary_col, spaceAfter=8, fontName="Helvetica-Bold")
    contact_style = ParagraphStyle('Contact', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=20)
    header_style = ParagraphStyle('Header', parent=styles['Heading2'], fontSize=12, textColor=accent_col, spaceBefore=15, spaceAfter=6, fontName="Helvetica-Bold", textTransform='uppercase')
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8, alignment=TA_JUSTIFY, textColor=colors.HexColor("#334155"))
    role_style = ParagraphStyle('Role', parent=styles['Normal'], fontSize=11, spaceBefore=6, textColor=primary_col, fontName="Helvetica-Bold")
    company_style = ParagraphStyle('Company', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#64748b"), fontName="Helvetica-Oblique")

    story = []
    
    name = data.get('full_name', 'CANDIDATE NAME').upper()
    contact = f"{data.get('email', '')} | {data.get('location', '')}"
    story.append(Paragraph(name, name_style))
    story.append(Paragraph(contact, contact_style))
    
    if data.get('links'):
        links_text = []
        for link in data.get('links', []):
            clean_url = link['url'].replace('https://', '').replace('http://', '').replace('www.', '')
            links_text.append(f"<b>{link['label']}:</b> {clean_url}")
        story.append(Paragraph("  |  ".join(links_text), body_style))
        
    story.append(get_line(480, "#e2e8f0"))
    story.append(Spacer(1, 15))

    if data.get('summary'):
        story.append(Paragraph("PROFESSIONAL SUMMARY", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        story.append(Paragraph(data.get('summary', ''), body_style))

    if data.get('achievements'):
        story.append(Paragraph("AWARDS & ACHIEVEMENTS", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        for ach in data.get('achievements', []):
            story.append(Paragraph(f"• {ach}", body_style))

    if data.get('projects'):
        story.append(Paragraph("KEY PROJECTS", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        for proj in data.get('projects', []):
            title = f"{proj.get('title', 'Project')} | {proj.get('tech', '')}"
            story.append(Paragraph(title, role_style))
            bullets = [ListItem(Paragraph(p, body_style)) for p in proj.get('bullets', [])]
            story.append(ListFlowable(bullets, bulletType='bullet', start='circle', leftIndent=15, bulletColor=accent_col))
            story.append(Spacer(1, 6))

    if data.get('experience'):
        story.append(Paragraph("RELEVANT EXPERIENCE", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        for job in data.get('experience', []):
            story.append(Paragraph(job.get('role', 'Role'), role_style))
            story.append(Paragraph(f"{job.get('company', '')} | {job.get('dates', '')}", company_style))
            bullets = [ListItem(Paragraph(p, body_style)) for p in job.get('bullets', [])]
            story.append(ListFlowable(bullets, bulletType='bullet', start='circle', leftIndent=15, bulletColor=accent_col))
            story.append(Spacer(1, 8))

    if data.get('education'):
        story.append(Paragraph("EDUCATION", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        for edu in data.get('education', []):
            story.append(Paragraph(f"{edu.get('degree', '')}", role_style))
            story.append(Paragraph(f"{edu.get('institution', '')} | {edu.get('dates', '')}", company_style))
            if edu.get('grade'):
                story.append(Paragraph(f"Grade: {edu.get('grade')}", body_style))
            if edu.get('modules'):
                mod_text = "<b>Key Modules:</b> " + ", ".join(edu['modules'])
                story.append(Paragraph(mod_text, body_style))
            story.append(Spacer(1, 8))

    if data.get('skills'):
        story.append(Paragraph("TECHNICAL SKILLS", header_style))
        story.append(get_line(480, "#f1f5f9"))
        story.append(Spacer(1, 6))
        skills_text = "  •  ".join(data.get('skills', []))
        story.append(Paragraph(skills_text, body_style))

    doc.build(story)
    return filename

def generate_cv_json(cv_text, job_desc, job_link):
    prompt = f"""
    You are a Fact-Obsessed CV Reconstructor.
    Job Link: {job_link}
    Job Description: {job_desc}
    
    CRITICAL RULES:
    1. **Proper Nouns:** Never delete a Company Name, Client Name, or Award Title.
    2. **URLs:** Extract ALL links (ArtStation, GitHub, Itch.io, LinkedIn) into 'links'.
    3. **Modules:** If user lists specific modules (e.g. "Level Design 304"), put them in 'modules'.
    4. **Projects:** Keep specific project names (e.g. "Neon Racer").
    5. **NO GENERIC "NO EXPERIENCE":** Convert University Projects into 'experience'.

    INPUT CV: {cv_text}
    
    OUTPUT JSON SCHEMA:
    {{
        "full_name": "String",
        "email": "String",
        "location": "String",
        "links": [ {{ "label": "Label", "url": "URL" }} ],
        "summary": "String",
        "achievements": ["String"],
        "projects": [ {{ "title": "Name", "tech": "Tools", "bullets": ["Action -> Result"] }} ],
        "experience": [ {{ "role": "Role", "company": "Company", "dates": "Date", "bullets": ["Action -> Result"] }} ],
        "education": [ {{ "degree": "Name", "institution": "Uni", "dates": "Date", "grade": "Grade", "modules": ["Mod1", "Mod2"] }} ],
        "skills": ["Skill1", "Skill2"]
    }}
    """
    
    # --- WATERFALL LOOP ---
    # We iterate through the VALID list. If one works, we return immediately.
    for model_name in VALID_MODELS:
        print(f"🔄 Trying model: {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            
            clean_text = response.text.replace('```json', '').replace('```', '')
            print(f"✅ SUCCESS! Generated with: {model_name}")
            return json.loads(clean_text)
            
        except Exception as e:
            # Catch 429 (Rate Limit), 404 (Not Found), etc. and continue
            print(f"⚠️ Failed ({model_name}): {e}")
            print("   -> Skipping to next model...")
            continue

    # If we exit the loop, everything failed
    print("❌ CRITICAL: All models failed.")
    return None

@app.route('/')
def index(): return render_template('landingpage.html')

@app.route('/upload')
def upload_cv(): return render_template('Cvupload.html')

@app.route('/results')
def results(): return render_template('results.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'cv' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['cv']
    job_desc = request.form.get('job_description', '')
    job_link = request.form.get('job_link', 'Not provided')
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        text = extract_cv_text(path)
        if os.path.exists(path): os.remove(path)
        
        if not text or len(text) < 10:
            return jsonify({'error': 'CV is empty or unreadable.'}), 400

        cv_data = generate_cv_json(text, job_desc, job_link)
        
        if not cv_data: 
            return jsonify({'error': 'All AI models are currently busy or rate-limited. Please wait 1 minute.'}), 500

        pdf_name = f"Improved_CV_{os.urandom(4).hex()}.pdf"
        create_pdf(cv_data, pdf_name)
        
        return jsonify({'success': True, 'pdf_url': f"/download/{pdf_name}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['GENERATED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)