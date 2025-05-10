from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
import os
import fitz  # PyMuPDF for PDF parsing

load_dotenv()

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def extract_text(file):
    if file.filename.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    else:
        return ""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        job_desc = request.form.get("job", "")
        resume = request.form.get("resume", "")
        language = request.form.get("language", "English")
        file = request.files.get("resumeFile")

        if file and file.filename:
            resume = extract_text(file)

        if not resume or not job_desc:
            return jsonify({"result": "Error: Resume or Job Description missing."})

        prompt = f"""
        Respond in {language}.
        Compare the following resume and job description.
        1. List key qualifications missing in the resume.
        2. Suggest 3 resume bullet points tailored to the job.
        3. Write a short, personalized cover letter based on the resume and job.

        Resume:
        {resume}

        Job Description:
        {job_desc}
        """

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        print("Error:", e)
        return jsonify({"result": f"An error occurred:\n\n{str(e)}"})

@app.route("/match-jobs", methods=["POST"])
def match_jobs():
    try:
        file = request.files.get("matchResume")
        if not file or not file.filename:
            return jsonify({"result": "Error: No resume uploaded."})

        resume = extract_text(file)
        prompt = f"""
        Based on the resume below, suggest 3 job titles and industries this candidate is well-suited for.

        Resume:
        {resume}
        """

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        print("Error:", e)
        return jsonify({"result": f"An error occurred:\n\n{str(e)}"})

@app.route("/rewrite-bullet", methods=["POST"])
def rewrite_bullet():
    try:
        bullet = request.form.get("bullet", "")
        if not bullet:
            return jsonify({"result": "Error: No bullet point provided."})

        prompt = f"""
        Improve the following resume bullet point. Make it stronger, clearer, and more results-oriented:

        Bullet Point:
        {bullet}
        """

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        print("Error:", e)
        return jsonify({"result": f"An error occurred:\n\n{str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
