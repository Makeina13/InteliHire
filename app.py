from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import re
import random
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import mm, cm

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

# Color schemes: ONLY 4 themes as specified
COLOR_SCHEMES = [
    {
        'name': 'Black + White + Grey',
        'sidebar_bg': '#2b2b2b',          # Dark grey sidebar (like Rory Hunter)
        'header_bg': '#e8e8e8',           # Light grey header box
        'header_border': '#000000',       # Black border for header
        'text_dark': '#1a1a1a',           # Dark text
        'text_medium': '#4a4a4a',         # Medium grey text
        'text_light': '#6a6a6a',          # Light grey text
        'line_color': '#000000',          # Black lines/accents
        'sidebar_text': '#FFFFFF'         # White text in sidebar
    },
    {
        'name': 'Navy Blue + White + Grey',
        'sidebar_bg': '#1B365D',          # Navy blue sidebar
        'header_bg': '#f0f4f8',           # Light blue-grey header
        'header_border': '#1B365D',       # Navy border
        'text_dark': '#1a1a1a',
        'text_medium': '#4a4a4a',
        'text_light': '#1B365D',          # Navy accents
        'line_color': '#1B365D',
        'sidebar_text': '#FFFFFF'
    },
    {
        'name': 'Forest Green + White + Grey',
        'sidebar_bg': '#1e5631',          # Forest green sidebar
        'header_bg': '#f0f8f0',           # Light green-grey header
        'header_border': '#228B22',       # Green border
        'text_dark': '#1a1a1a',
        'text_medium': '#4a4a4a',
        'text_light': '#228B22',          # Green accents
        'line_color': '#228B22',
        'sidebar_text': '#FFFFFF'
    },
    {
        'name': 'Purple + White + Grey',
        'sidebar_bg': '#4a3a5f',          # Purple sidebar
        'header_bg': '#f5f0f8',           # Light purple-grey header
        'header_border': '#6B4C9A',       # Purple border
        'text_dark': '#1a1a1a',
        'text_medium': '#4a4a4a',
        'text_light': '#6B4C9A',          # Purple accents
        'line_color': '#6B4C9A',
        'sidebar_text': '#FFFFFF'
    }
]

CADENCE_PRESETS = {
    'low': {
        'temperature': 1.0,
        'top_p': 0.95,
        'description': "Direct and factual. Short sentences. Specific numbers."
    },
    'medium': {
        'temperature': 1.2,
        'top_p': 0.97,
        'description': "Natural voice. Mix of sentence lengths. Mini-stories."
    },
    'high': {
        'temperature': 1.4,
        'top_p': 0.98,
        'description': "Distinct personal voice. Fragments. Asides. Opinions."
    }
}

def normalize_cadence(raw_value):
    value = (raw_value or 'medium').strip().lower()
    return value if value in CADENCE_PRESETS else 'medium'

def get_generation_config(cadence):
    preset = CADENCE_PRESETS[cadence]
    return genai.GenerationConfig(
        temperature=preset['temperature'],
        top_p=preset['top_p'],
        top_k=50,
        response_mime_type="application/json"
    )

def get_prioritized_model_list():
    print("--------------------------------------------------")
    print("📋 BUILDING MODEL LIST...")
    
    try:
        all_my_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace('models/', '')
                all_my_models.append(name)
        
        prioritized = []
        
        p1 = [m for m in all_my_models if 'flash' in m and 'lite' in m]
        p1.sort(key=lambda x: 'preview' in x) 
        prioritized.extend(p1)

        p2 = [m for m in all_my_models if 'gemini-2.5-flash' in m and m not in prioritized]
        prioritized.extend(p2)

        p3 = [m for m in all_my_models if 'gemini-2.0-flash' in m and m not in prioritized]
        prioritized.extend(p3)
        
        for m in all_my_models:
            if m not in prioritized:
                if 'robotics' not in m:
                    prioritized.append(m)
                
        print(f"✅ Found {len(prioritized)} valid models.")
        if len(prioritized) > 0:
            print(f"🚀 Top Pick (Priority 1): {prioritized[0]}")
        print("--------------------------------------------------")
        return prioritized

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return ['gemini-2.0-flash-lite-001', 'gemini-2.5-flash', 'gemini-2.0-flash']

VALID_MODELS = get_prioritized_model_list()

AI_PATTERNS = [
    (r'\bplayed a key role in\b', 'helped with'),
    (r'\btook initiative in\b', ''),
    (r'\bensuring\b', 'keeping'),
    (r'\bfostering\b', 'building'),
    (r'\benhancing\b', 'improving'),
    (r'\bimplementing\b', 'setting up'),
    (r'\butilizing\b', 'using'),
    (r'\bleveraging\b', 'using'),
    (r'\bfacilitating\b', 'running'),
    (r'\bspearheading\b', 'leading'),
    (r'\bdemonstrating\b', 'showing'),
    (r'\bcollaborative\b', 'team'),
    (r'\binnovative\b', 'new'),
    (r'\bdynamic\b', ''),
    (r'\bstrategic\b', ''),
    (r'\bcomprehensive\b', 'full'),
    (r'\bpassionate about\b', 'interested in'),
    (r'\bdedicated\b', ''),
    (r'\bdriven\b', ''),
    (r'\bmotivated\b', ''),
    (r'\benthusiastic\b', 'keen'),
    (r', resulting in\b', '. This led to'),
    (r', directly resulting in\b', '. Got'),
    (r'the planning and delivery of\b', 'running'),
    (r'\bacted as the primary\b', 'was the'),
]

def clean_ai_patterns(text):
    if not text or not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in AI_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'\s+\.', '.', result)
    result = re.sub(r'\s+,', ',', result)
    return result

def clean_cv_data(cv_data):
    if not cv_data:
        return cv_data
    
    if cv_data.get('summary'):
        cv_data['summary'] = clean_ai_patterns(cv_data['summary'])
    
    for job in cv_data.get('experience', []):
        if job.get('bullets'):
            job['bullets'] = [clean_ai_patterns(b) for b in job['bullets']]
    
    for proj in cv_data.get('projects', []):
        if proj.get('bullets'):
            proj['bullets'] = [clean_ai_patterns(b) for b in proj['bullets']]
    
    if cv_data.get('achievements'):
        cv_data['achievements'] = [clean_ai_patterns(a) for a in cv_data['achievements']]
    
    return cv_data

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

def create_pdf(data, filename):
    # Import BaseDocTemplate and Frame locally to fix the padding issue
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Flowable

    # Random color scheme - ONE theme per PDF
    color_scheme = random.choice(COLOR_SCHEMES)
    print(f"🎨 Using color scheme: {color_scheme['name']}")
    
    filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
    
    page_width, page_height = A4
    sidebar_width = 180 
    main_width = page_width - sidebar_width
    
    # 1. HELPER: Convert Hex to Alpha Color
    def hex_to_alpha(hex_code, alpha=0.15):
        hex_code = hex_code.lstrip('#')
        return colors.Color(
            int(hex_code[0:2], 16)/255.0,
            int(hex_code[2:4], 16)/255.0,
            int(hex_code[4:6], 16)/255.0,
            alpha=alpha
        )

    # Colors
    sidebar_bg = colors.HexColor(color_scheme['sidebar_bg'])
    sidebar_text = colors.HexColor(color_scheme['sidebar_text'])
    text_dark = colors.HexColor(color_scheme['text_dark'])
    text_medium = colors.HexColor(color_scheme['text_medium'])
    text_light = colors.HexColor(color_scheme['text_light'])
    header_border = colors.HexColor(color_scheme['header_border'])
    
    # Kept alpha=0.25 as per previous successful adjustment
    header_bg_transparent = hex_to_alpha(color_scheme['header_border'], alpha=0.25)
    
    # 2. DOCUMENT SETUP (Using BaseDocTemplate for Zero Padding)
    doc = BaseDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=0,
        leftMargin=0,
        topMargin=0,
        bottomMargin=0
    )

    # 3. BACKGROUND DRAWING (Sidebar)
    def draw_sidebar_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(sidebar_bg)
        canvas.rect(0, 0, sidebar_width, page_height, stroke=0, fill=1)
        canvas.restoreState()

    # 4. CUSTOM FRAME (Explicitly 0 padding to fix the gap)
    full_frame = Frame(
        0, 0, page_width, page_height,
        leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0, 
        id='normal'
    )
    
    full_page_template = PageTemplate(id='full_page', frames=full_frame, onPage=draw_sidebar_bg)
    doc.addPageTemplates([full_page_template])

    styles = getSampleStyleSheet()

    # === STYLE DEFINITIONS ===
    name_style = ParagraphStyle('NameStyle', parent=styles['Normal'], fontSize=28, fontName="Helvetica-Bold", textColor=text_dark, alignment=TA_CENTER, spaceAfter=4, leading=34)
    job_subtitle_style = ParagraphStyle('JobSubtitle', parent=styles['Normal'], fontSize=11, textColor=text_medium, alignment=TA_CENTER, spaceAfter=0, leading=14)
    
    # Sidebar Section: Removed indent (0) so it relies on box padding
    sidebar_section_style = ParagraphStyle(
        'SidebarSection', 
        parent=styles['Normal'], 
        fontSize=11, 
        fontName="Helvetica-Bold", 
        textColor=sidebar_text, 
        alignment=TA_LEFT, 
        leftIndent=0, 
        rightIndent=0, 
        leading=16
    )
    
    # Sidebar Content: Added leftIndent=8 to align with Bullets and Header Padding
    sidebar_label = ParagraphStyle('SidebarLabel', parent=styles['Normal'], fontSize=9, fontName="Helvetica-Bold", textColor=sidebar_text, spaceBefore=6, spaceAfter=1, leftIndent=8)
    sidebar_value = ParagraphStyle('SidebarValue', parent=styles['Normal'], fontSize=9, textColor=sidebar_text, spaceAfter=4, leading=11, leftIndent=8)
    sidebar_bullet = ParagraphStyle('SidebarBullet', parent=styles['Normal'], fontSize=9, textColor=sidebar_text, leftIndent=8, spaceAfter=3, leading=11)
    
    main_section_style = ParagraphStyle('MainSection', parent=styles['Normal'], fontSize=12, fontName="Helvetica-Bold", textColor=text_dark, alignment=TA_LEFT, leftIndent=10, rightIndent=10, leading=18)
    profile_text = ParagraphStyle('ProfileText', parent=styles['Normal'], fontSize=10, textColor=text_medium, leading=14, spaceAfter=10, alignment=TA_LEFT)
    date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=9, textColor=text_light, spaceBefore=8, spaceAfter=2)
    position_style = ParagraphStyle('PositionStyle', parent=styles['Normal'], fontSize=11, fontName="Helvetica-Bold", textColor=text_dark, spaceAfter=1, leading=13)
    company_style = ParagraphStyle('CompanyStyle', parent=styles['Normal'], fontSize=10, textColor=text_medium, spaceAfter=4)
    bullet_style = ParagraphStyle('BulletStyle', parent=styles['Normal'], fontSize=9, textColor=text_medium, leftIndent=10, leading=12, spaceAfter=3)
    
    # Education Styles: Added leftIndent=8
    edu_degree = ParagraphStyle('EduDegree', parent=styles['Normal'], fontSize=10, fontName="Helvetica-Bold", textColor=sidebar_text, spaceBefore=4, spaceAfter=1, leftIndent=8)
    edu_detail = ParagraphStyle('EduDetail', parent=styles['Normal'], fontSize=9, textColor=sidebar_text, spaceAfter=2, leftIndent=8)

    # === CUSTOM FLOWABLE: Name Box with Job Gap ===
    class NameJobFrame(Flowable):
        def __init__(self, name_text, job_text, name_style, job_style, border_color, bg_color, padding=15):
            Flowable.__init__(self)
            self.name_text = name_text
            self.job_text = job_text
            self.name_style = name_style
            self.job_style = job_style
            self.border_color = border_color
            self.bg_color = bg_color
            self.padding = padding
            self.line_width = 1.5
            
        def wrap(self, availWidth, availHeight):
            self.name_p = Paragraph(self.name_text, self.name_style)
            self.job_p = Paragraph(self.job_text, self.job_style)
            
            # Calculate how much space the text actually needs
            w_name, h_name = self.name_p.wrap(availWidth, availHeight)
            w_job, h_job = self.job_p.wrap(availWidth, availHeight)
            
            self.box_height = h_name + 2 * self.padding
            self.job_height = h_job
            
            # --- CHANGE IS HERE ---
            # Old: self.width = availWidth (This forced full width)
            # New: Set width to the name's text width + padding
            self.width = w_job + -40
            
            # Optional: Ensure the box is at least as wide as the Job Title (+ margin)
            # if (w_job + 30) > self.width:
            #     self.width = w_job + 30

            self.height = self.box_height + (h_job / 2)
            return self.width, self.height

        def draw(self):
            # Coordinates:
            box_bottom = self.job_height / 2
            box_top = box_bottom + self.box_height
            
            # 1. Draw Background (Box only)
            self.canv.saveState()
            self.canv.setFillColor(self.bg_color)
            self.canv.rect(0, box_bottom, self.width, self.box_height, stroke=0, fill=0)
            
            # 2. Draw Name
            # --- ADD THIS LINE ---
            # Re-wrap the name to the exact box width (minus padding) so 'CENTER' works correctly
            self.name_p.wrap(self.width - 2*self.padding, self.box_height)
            
            # (Keep this line the same)
            self.name_p.drawOn(self.canv, self.padding, box_bottom + self.padding)
            
            # 3. Draw Job Title
            if self.job_text:
                # --- ADD THIS LINE ---
                # Re-wrap job title to the full box width so it centers exactly on the border
                self.job_p.wrap(self.width, self.job_height)
                
                # (Keep this line the same)
                self.job_p.drawOn(self.canv, 0, 0)
                
                # Calculate gap width
                job_width = stringWidth(self.job_text, self.job_style.fontName, self.job_style.fontSize)
                gap_w = job_width + 20 
            else:
                gap_w = 0

            # 4. Draw Borders (The rest stays the same)
            self.canv.setStrokeColor(self.border_color)
            self.canv.setLineWidth(self.line_width)
            
            # Top, Left, Right
            self.canv.line(0, box_top, self.width, box_top) # Top
            self.canv.line(0, box_bottom, 0, box_top)       # Left
            self.canv.line(self.width, box_bottom, self.width, box_top) # Right
            
            # Bottom Line (Split for Gap)
            if gap_w > 0:
                mid_x = self.width / 2
                gap_start = mid_x - (gap_w / 2)
                gap_end = mid_x + (gap_w / 2)
                
                self.canv.line(0, box_bottom, gap_start, box_bottom)
                self.canv.line(gap_end, box_bottom, self.width, box_bottom)
            else:
                self.canv.line(0, box_bottom, self.width, box_bottom)
                
            self.canv.restoreState()

    # Helper: Boxed Header
    def make_boxed_header(text, style, border_color, bg_color=None, fit_width=False):
        para = Paragraph(text, style)
        
        if fit_width:
            col_width = 135 
            
            # Tighter style for sidebar labels
            table_style = [
                ('BOX', (0, 0), (-1, -1), 0.75, border_color),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]
        else:
            # Original logic for main content
            col_width = style.leftIndent + style.rightIndent + 350
            table_style = [
                ('BOX', (0, 0), (-1, -1), 1.5, border_color),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]
        header_table = Table([[para]], colWidths=[col_width])
        if bg_color:
            table_style.append(('BACKGROUND', (0, 0), (-1, -1), bg_color))
        
        header_table.setStyle(TableStyle(table_style))
        return header_table

    # === BUILD SIDEBAR CONTENT ===
    sidebar_elements = []
    sidebar_elements.append(Spacer(1, 15))
    
    # Use fit_width=True for sidebar headers
    sidebar_elements.append(make_boxed_header("Profile", sidebar_section_style, sidebar_text, fit_width=True))
    sidebar_elements.append(Spacer(1, 8))
    
    if data.get('email'):
        sidebar_elements.append(Paragraph("✉ Email", sidebar_label))
        sidebar_elements.append(Paragraph(data.get('email', ''), sidebar_value))
    
    if data.get('location'):
        sidebar_elements.append(Paragraph("☛ Location", sidebar_label))
        sidebar_elements.append(Paragraph(data.get('location', ''), sidebar_value))
    
    if data.get('links'):
        for link in data.get('links', []):
            clean_url = link['url'].replace('https://', '').replace('http://', '').replace('www.', '')
            sidebar_elements.append(Paragraph(f"✎ {link.get('label', 'Link')}", sidebar_label))
            sidebar_elements.append(Paragraph(clean_url, sidebar_value))
    
    if data.get('skills'):
        sidebar_elements.append(Spacer(1, 12))
        sidebar_elements.append(make_boxed_header("Skills", sidebar_section_style, sidebar_text, fit_width=True))
        sidebar_elements.append(Spacer(1, 6))
        
        for skill in data.get('skills', []):
            if skill:
                sidebar_elements.append(Paragraph(f" {skill}", sidebar_bullet))
    
    if data.get('education'):
        sidebar_elements.append(Spacer(1, 12))
        sidebar_elements.append(make_boxed_header("Education", sidebar_section_style, sidebar_text, fit_width=True))
        sidebar_elements.append(Spacer(1, 6))
        
        for edu in data.get('education', []):
            degree = edu.get('degree', '')
            institution = edu.get('institution', '')
            dates = edu.get('dates', '')
            grade = edu.get('grade', '')
            
            if degree:
                sidebar_elements.append(Paragraph(f"<b>{degree}</b>", edu_degree))
            if institution:
                sidebar_elements.append(Paragraph(institution, edu_detail))
            if dates:
                sidebar_elements.append(Paragraph(dates, edu_detail))
            if grade and grade.lower() not in ['n/a', 'na', '']:
                sidebar_elements.append(Paragraph(f"Grade: {grade}", edu_detail))
            sidebar_elements.append(Spacer(1, 5))
    
    if data.get('achievements'):
        achievements = [a for a in data.get('achievements', []) if a]
        if achievements:
            sidebar_elements.append(Spacer(1, 12))
            sidebar_elements.append(make_boxed_header("Achievements", sidebar_section_style, sidebar_text, fit_width=True))
            sidebar_elements.append(Spacer(1, 6))
            
            for ach in achievements:
                sidebar_elements.append(Paragraph(f" {ach}", sidebar_bullet))

    # === BUILD MAIN CONTENT ===
    # 1. Header (Custom NameJobFrame)
    name = data.get('full_name', 'Your Name')
    job_title = data.get('job_title', '')
    
    header_content = []
    header_content.append(Spacer(1, 15))

    name_job_box = NameJobFrame(name, job_title, name_style, job_subtitle_style, header_border, header_bg_transparent)
    header_content.append(name_job_box)
    
    header_content.append(Spacer(1, 16))
    
    header_inner = [[elem] for elem in header_content]
    header_table = Table(header_inner, colWidths=[main_width])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), header_bg_transparent),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    # 2. Body
    body_elements = []
    body_elements.append(Spacer(1, 16))

    if data.get('summary'):
        body_elements.append(Paragraph(data.get('summary', ''), profile_text))
        body_elements.append(Spacer(1, 12))
    
    if data.get('experience'):
        body_elements.append(make_boxed_header("Professional experience", main_section_style, header_border))
        body_elements.append(Spacer(1, 10))
        for job in data.get('experience', []):
            dates = job.get('dates', '')
            role = job.get('role', '')
            company = job.get('company', '')
            if dates: body_elements.append(Paragraph(dates, date_style))
            if role: body_elements.append(Paragraph(role, position_style))
            if company: body_elements.append(Paragraph(company, company_style))
            for bullet in job.get('bullets', []): body_elements.append(Paragraph(f"- {bullet}", bullet_style))
            body_elements.append(Spacer(1, 10))
            
    if data.get('projects'):
        projects = [p for p in data.get('projects', []) if p.get('title')]
        if projects:
            body_elements.append(Spacer(1, 8))
            body_elements.append(make_boxed_header("Projects", main_section_style, header_border))
            body_elements.append(Spacer(1, 10))
            for proj in projects:
                title = proj.get('title', '')
                tech = proj.get('tech', '')
                if title: body_elements.append(Paragraph(title, position_style))
                if tech: body_elements.append(Paragraph(tech, company_style))
                for bullet in proj.get('bullets', []): body_elements.append(Paragraph(f"• {bullet}", bullet_style))
                body_elements.append(Spacer(1, 10))

    # Body Table (Padded)
    body_inner = [[elem] for elem in body_elements]
    body_table = Table(body_inner, colWidths=[main_width])
    body_table.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 0), 
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    # Main Stack (Header + Body)
    main_stack = Table([[header_table], [body_table]], colWidths=[main_width])
    main_stack.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), header_bg_transparent),
        ('LEFTPADDING', (0, 0), (-1, -1), 0), 
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0), 
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    # === FINAL LAYOUT ===
    sidebar_inner = [[elem] for elem in sidebar_elements]
    sidebar_table = Table(sidebar_inner, colWidths=[sidebar_width])
    sidebar_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    # Master Table
    layout_data = [[sidebar_table, main_stack]]
    layout_table = Table(layout_data, colWidths=[sidebar_width, main_width])
    layout_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    story = [layout_table]
    doc.build(story)
    return filename


def parse_list_field(raw_value):
    if not raw_value:
        return []
    parts = re.split(r'[,;\n]+', raw_value)
    return [p.strip() for p in parts if p.strip()]

def parse_voice_bank(form):
    return {
        'voice_phrases': parse_list_field(form.get('voice_phrases', '')),
        'opinions': parse_list_field(form.get('opinions', '')),
        'metaphors': parse_list_field(form.get('metaphors', '')),
        'aside': form.get('aside', '').strip(),
        'humour_allowed': form.get('humour_allowed', 'false').strip().lower() in {'true', '1', 'yes', 'on'}
    }

def build_prompt(cv_text, job_desc, job_link, cadence, voice_bank):
    cadence_desc = CADENCE_PRESETS[cadence]['description']
    
    voice_context = ""
    if voice_bank.get('aside'):
        voice_context = f"\nCandidate background: {voice_bank['aside']}"

    return f"""You are a human CV writer. Write like a person, not an AI.

JOB: {job_desc}
{voice_context}

## STEP 1: UNDERSTAND WHY AI TEXT GETS DETECTED

AI writes like this (NEVER DO THIS):
- "Played a key role in enhancing X by fostering Y"
- "Took initiative in resolving issues, ensuring smooth operations"  
- "Boosted revenue by implementing strategies"
- "Developed and delivered focused talks that increased engagement"
- "Acted as the primary advocate, directly resulting in..."
- "Co-managed the planning and delivery of the programme"

These get flagged because:
1. "[Past verb] + [noun] + by + [gerund]" is an AI pattern
2. "resulting in", "ensuring", "fostering" are AI words
3. Every sentence has the same polished structure
4. No personality, just template filling

## STEP 2: WRITE LIKE THIS INSTEAD

SUMMARY (first person, semi-casual but professional):

EXPERIENCE BULLETS (tell mini-stories, vary length):


## STEP 3: STRUCTURAL RULES

1. Bullet lengths MUST vary:
   - One bullet: 20-30 words (the story)
   - Next bullet: 5-12 words (just the fact)
   - Next bullet: 15-20 words (medium with context)

2. Start bullets differently:
   - "I..." / "Got..." / "Ran..." / "Fixed..." / "Helped..."
   - "Our team..." / "The project..." / "That summer..."
   - Never start two bullets the same way

3. Include ONE of these human elements per job:
   - A short fragment: "Nothing major."
   - A parenthetical: "—not as easy as it sounds"
   - An honest qualifier: "probably", "about", "roughly"
   - A mild opinion: "Worked really well"

4. BANNED WORDS (if you use these, you fail):
   enhancing, fostering, implementing, ensuring, utilizing
   leveraging, spearheading, facilitating, demonstrating
   collaborative, innovative, dynamic, strategic
   passionate, dedicated, driven,
   "played a key role", "took initiative", "resulting in"
   "the planning and delivery of"

5. Do not leave more than two spaces in between each word 

6. Do not use any "-" 



## INPUT CV:
{cv_text}

## OUTPUT (JSON only, no explanation):
{{
    "full_name": "string",
    "email": "string",
    "location": "string", 
    "links": [{{"label": "string", "url": "string"}}],
    "summary": "First person. 3-4 sentences. Contractions. sounds professional but not robotic",
    "achievements": ["string"],
    "projects": [{{"title": "string", "tech": "string", "bullets": ["varied length human sentences"]}}],
    "experience": [{{"role": "string", "company": "string", "dates": "string", "bullets": ["MUST vary in length and structure"]}}],
    "education": [{{"degree": "string", "institution": "string", "dates": "string", "grade": "string", "modules": ["string"]}}],
    "skills": ["string"]

    IMPORTANT: if any information is missing from this list leave the space blank 
}}"""

def generate_cv_json(cv_text, job_desc, job_link, cadence='medium', voice_bank=None):
    voice_bank = voice_bank or {}
    prompt = build_prompt(cv_text, job_desc, job_link, cadence, voice_bank)
    generation_config = get_generation_config(cadence)
    
    for model_name in VALID_MODELS:
        print(f"🔄 Trying model: {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            
            clean_text = response.text.replace('```json', '').replace('```', '')
            print(f"✅ SUCCESS! Generated with: {model_name}")
            cv_data = json.loads(clean_text)
            cv_data = clean_cv_data(cv_data)
            return cv_data
            
        except Exception as e:
            print(f"⚠️ Failed ({model_name}): {e}")
            print("   -> Skipping to next model...")
            continue

    print("❌ CRITICAL: All models failed.")
    return None

VAGUE_PHRASES = [
    "improved efficiency",
    "increased efficiency",
    "increased sales",
    "boosted sales",
    "streamlined",
    "optimized",
    "responsible for",
    "worked on",
    "handled",
    "assisted with",
    "helped",
    "supported"
]

MONTH_TOKENS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

def bullet_has_anchor(text, skills_lower):
    if not text:
        return False
    if re.search(r"[0-9%£$€]", text):
        return True
    lower = text.lower()
    if any(month in lower for month in MONTH_TOKENS):
        return True
    if any(skill in lower for skill in skills_lower if len(skill) > 2):
        return True
    if re.search(r"\b[A-Z]{2,}\b", text):
        return True
    return False

def generate_quality_report(cv_data):
    if not cv_data:
        return []
    skills_lower = [s.lower() for s in cv_data.get('skills', []) if isinstance(s, str)]
    issues = []

    def inspect_bullets(bullets, context):
        for bullet in bullets or []:
            if not isinstance(bullet, str):
                continue
            lower = bullet.lower()
            for phrase in VAGUE_PHRASES:
                if phrase in lower and not bullet_has_anchor(bullet, skills_lower):
                    issues.append({
                        "context": context,
                        "bullet": bullet,
                        "issue": f"Phrase '{phrase}' lacks a supporting metric, tool, or date."
                    })
                    break

    for idx, job in enumerate(cv_data.get('experience', [])):
        inspect_bullets(job.get('bullets', []), f"experience[{idx}]")
    for idx, project in enumerate(cv_data.get('projects', [])):
        inspect_bullets(project.get('bullets', []), f"projects[{idx}]")

    return issues

@app.route('/')
def index(): return render_template('landingpage.html')

@app.route('/upload')
def upload_cv(): return render_template('Cvupload.html')

@app.route('/results')
def results(): return render_template('results.html')

@app.route('/subscriptions')
def subscriptions(): return render_template('subscriptions.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. We still accept the file upload so the frontend doesn't crash
    if 'cv' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['cv']
    
    # 2. We skip extraction and AI generation entirely
    print("⚠️ MOCK MODE: Skipping AI, using dummy data for layout testing.")

    # 3. Define the Dummy Data
    cv_data = {
        "full_name": "Mia Smith",
        "job_title": "Cheif Executive Officer",
        "email": "leonardofrazer@yahoo.co.uk",
        "location": "Liverpool, UK",
        "links": [
            {"label": "Portfolio", "url": "editsbylennox.my.canva.site/vfx-portfolio"},
        ],
        "summary": "I'm a video editor who really enjoys making short, punchy content that grabs attention right away. I'm good at working with others to nail a creative vision.",
        "skills": [
            "Video Editing", "Graphic Design", "Creative Strategy", 
            "Adobe Premiere Pro", "After Effects", "DaVinci Resolve"
        ],
        "experience": [
            {
                "role": "Video Editor (Freelance)",
                "company": "Self-Employed",
                "dates": "Jan 2022 - Present",
                "bullets": [
                    "Worked closely with car dealerships to figure out exactly what their brand looked like.",
                    "Got raw footage looking top-notch using color grading.",
                    "Helped out with podcast videos for creators like Iman Ghazi."
                ]
            },
            {
                "role": "Junior Editor",
                "company": "Creative Agency London",
                "dates": "2020 - 2021",
                "bullets": [
                    "Assisted senior editors with rough cuts and timeline management.",
                    "Organized terabytes of footage for quick access."
                ]
            }
        ],
        "education": [
            {
                "degree": "BSC Game Design",
                "institution": "University of Liverpool",
                "dates": "2023-2026",
                "grade": "In Progress"
            },
            {
                "degree": "Level 3 in Game Design",
                "institution": "Carshalton College",
                "dates": "2021-2023",
                "grade": "Distinction"
            }
        ],
        "achievements": [
            "Duke Of Edinburgh Award - Bronze",
            "Video Editing Competition - Sutra Edits"
        ]
    }

    # 4. Create the PDF using the dummy data
    filename = secure_filename(file.filename)
    base_name = os.path.splitext(filename)[0]
    pdf_name = f"{base_name}_TEST_LAYOUT.pdf"  

    try:
        # Pass the dummy data to your create_pdf function
        create_pdf(cv_data, pdf_name)
        quality_report = generate_quality_report(cv_data)
        
        # Return success (frontend thinks AI did it)
        return jsonify({
            'success': True,
            'pdf_url': f"/download/{pdf_name}",
            'quality_report': []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['GENERATED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)