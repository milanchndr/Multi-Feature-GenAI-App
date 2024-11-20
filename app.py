import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')
app.config["UPLOAD_FOLDER"] = "./uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt"}

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Few-shot prompt templates
FEW_SHOT_PROMPTS = {
    "summarize": {
        "examples": [
            {
                "input": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "output": "Key Points:\n• Machine learning is a data analysis method within artificial intelligence\n• Focuses on systems learning and identifying patterns automatically\n• Minimizes the need for human intervention in analytical processes"
            },
            {
                "input": "Climate change is a long-term shift in global or regional climate patterns. It involves changes in temperature, precipitation, and extreme weather events, primarily driven by human activities like greenhouse gas emissions.",
                "output": "Key Points:\n• Climate change represents significant long-term shifts in global weather patterns\n• Involves changes in temperature, precipitation, and extreme weather\n• Primarily attributed to human-induced greenhouse gas emissions"
            }
        ],
        "instruction": "Provide a concise summary with clear, bullet-pointed key points."
    },
    "feedback": {
        "examples": [
            {
                "input": "The research paper discusses the impact of social media on mental health among teenagers. It explores various psychological effects and suggests potential mitigation strategies.",
                "output": "Feedback:\n• Structure: Clear progression from problem identification to potential solutions\n• Clarity: Technical terms are well-explained, making the content accessible\n• Completeness: Covers multiple dimensions of the social media-mental health relationship"
            }
        ],
        "instruction": "Provide constructive feedback focusing on structure, clarity, and completeness in a concise, professional manner."
    },
    "extract": {
        "examples": [
            {
                "input": "A comprehensive study on renewable energy implementation in urban environments. The research examines solar and wind energy potential in metropolitan areas, analyzing infrastructure challenges and economic feasibility. Key objectives include identifying sustainable energy solutions and reducing carbon emissions.",
                "output": "Key Sections:\n• Problem Statement: Urban energy sustainability and carbon emission reduction\n• Objectives: Identify renewable energy potential in metropolitan settings\n• Methodology: Analyzing solar and wind energy infrastructure and economic viability\n• Conclusions: Potential for implementable sustainable urban energy solutions"
            }
        ],
        "instruction": "Extract key sections with clear, concise descriptions."
    },
    "questions": {
        "examples": [
            {
                "input": "Artificial Intelligence represents a transformative technology that combines machine learning, neural networks, and advanced algorithms to create systems capable of performing complex cognitive tasks.",
                "output": "Viva/Presentation Questions:\n1. How do neural networks differ from traditional computing approaches?\n2. What are the ethical considerations in AI development?\n3. Explain the role of machine learning in AI system evolution\n4. Discuss potential limitations of current AI technologies"
            }
        ],
        "instruction": "Generate thought-provoking, academically relevant questions."
    }
}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def generate_response(prompt, feature):
    """Generate a response using Gemini API with few-shot prompting."""
    try:
        # Construct few-shot prompt
        few_shot_context = "\n\n".join([
            f"Example Input: {ex['input']}\nExample Output: {ex['output']}"
            for ex in FEW_SHOT_PROMPTS[feature]['examples']
        ])
        
        full_prompt = (
            f"{FEW_SHOT_PROMPTS[feature]['instruction']}\n\n"
            f"Few-shot Examples:\n{few_shot_context}\n\n"
            f"New Input:\n{prompt}"
        )

        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"An error occurred: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def home():
    """Main route to handle file/text processing and AI features."""
    content = None
    output = None
    feature = request.form.get("feature")

    if request.method == "POST":
        pasted_text = request.form.get("pasted_text")
        file = request.files.get("file")

        # Determine content source
        if pasted_text:
            content = pasted_text
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            
            if file.filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            else:
                content = file.read().decode('utf-8')

        # Process content based on selected feature
        if content and feature in FEW_SHOT_PROMPTS:
            prompt = f"{content}"
            output = generate_response(prompt, feature)

    return render_template("index.html", content=content, output=output)

if __name__ == "__main__":
    app.run(debug=True)