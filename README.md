InteliHire – AI Interview & CV Helper


This is a little project we are building to help everyone land a job and get better at interviews 

*InteliHire lets you:*

upload your CV (PDF/DOCX)

paste in a job description

get AI-powered feedback and a rewritten CV tailored to that role

It’s still very much a work-in-progress, but the plan is to eventually push this live so people can actually use it, not just look at the code on GitHub.

What it does right now

 Upload a CV and extract the text (PDF & DOCX supported)

 Send the CV + job description to Google Gemini for analysis

 Get a suitability check + improved CV content back

 Has a simple landing page + a separate CV upload page with nicer CSS and a glowing logo because… why not

    Tech stack
    
    Python + Flask
    
    Google Gemini API for the AI stuff
    
    HTML/CSS + a bit of JS for the frontend
    
    How to run it
    
    Clone the repo, install deps, and run Flask:
    
    pip install -r requirements.txt
    python app.py
    
    
    Then open your browser at http://127.0.0.1:5000/.
    
    You’ll also need a GEMINI_API_KEY in a .env file:
    
    GEMINI_API_KEY=your_api_key_here

Future plans: 

Add a live camera interview with an AI that wull ask questions based on the role you are looking for 

give feedback from that interview with a score rating and information on how to do better 

add rate limiting 

add authentication

add HTTPS security 

add input valdation 

*and more*
