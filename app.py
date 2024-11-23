import os
import difflib
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import glob


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')
app.config["UPLOAD_FOLDER"] = "./uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
WINSTON_API_URL = "https://api.gowinston.ai/v2/plagiarism"
WINSTON_API_KEY = os.getenv("WINSTON_API_KEY")

FEW_SHOT_PROMPTS = {
    "summarize": {
        "examples": [
            {
                "input": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
                "output": "Key Points:\n• Machine learning is a method of automated data analysis\n• Utilizes artificial intelligence for pattern recognition and decision-making\n• Reduces reliance on human intervention in analytical processes"
            },
            {
                "input": "Climate change is a long-term shift in global or regional climate patterns. It involves changes in temperature, precipitation, and extreme weather events, primarily driven by human activities like greenhouse gas emissions.",
                "output": "Key Points:\n• Climate change entails significant, long-term changes in weather patterns\n• Includes variations in temperature, precipitation, and extreme weather events\n• Largely caused by human activities, such as greenhouse gas emissions"
            }
        ],
        "instruction": "Provide a concise summary with clear, bullet-pointed key points that capture the core ideas and significance of the content."
    },
    "feedback": {
        "examples": [
            {
                "input": "The research paper discusses the impact of social media on mental health among teenagers. It explores various psychological effects and suggests potential mitigation strategies.",
                "output": "Feedback:\n• Structure: The paper flows logically from problem identification to solution proposals\n• Clarity: Ideas are well-articulated, with technical terms adequately defined\n• Completeness: The study addresses multiple dimensions of social media's psychological impact on teenagers"
            },
            {
                "input": "This report investigates the feasibility of autonomous vehicles in urban settings. It discusses technical challenges, regulatory hurdles, and potential societal benefits.",
                "output": "Feedback:\n• Structure: The report transitions smoothly across challenges, regulations, and benefits\n• Clarity: Technical terms and concepts are clearly explained, ensuring accessibility\n• Completeness: Key factors influencing urban adoption of autonomous vehicles are thoroughly examined"
            }
        ],
        "instruction": "Provide constructive, detailed feedback emphasizing structure, clarity, and completeness in a professional tone."
    },
    "extract": {
        "examples": [
            {
                "input": "A comprehensive study on renewable energy implementation in urban environments. The research examines solar and wind energy potential in metropolitan areas, analyzing infrastructure challenges and economic feasibility. Key objectives include identifying sustainable energy solutions and reducing carbon emissions.",
                "output": "Key Sections:\n• Problem Statement: Addressing urban energy sustainability and reducing carbon emissions\n• Objectives: Evaluate solar and wind energy viability in metropolitan areas\n• Methodology: Assessing infrastructure and economic challenges\n• Conclusions: Highlighting implementable sustainable energy solutions for cities"
            },
            {
                "input": "This study explores the use of blockchain in supply chain management, focusing on enhancing transparency, reducing fraud, and improving efficiency. Key challenges like scalability and regulatory compliance are also discussed.",
                "output": "Key Sections:\n• Problem Statement: Addressing transparency and fraud in supply chain management\n• Objectives: Improve efficiency through blockchain implementation\n• Methodology: Analyze scalability and regulatory compliance challenges\n• Conclusions: Potential of blockchain to transform supply chain operations"
            }
        ],
        "instruction": "Extract and organize the key sections (e.g., problem statement, objectives, methodology, conclusions) in a concise format."
    },
    "questions": {
        "examples": [
            {
                "input": "Artificial Intelligence represents a transformative technology that combines machine learning, neural networks, and advanced algorithms to create systems capable of performing complex cognitive tasks.",
                "output": "Viva/Presentation Questions:\n1. How do neural networks differ from traditional algorithms?\n2. What are the key challenges in implementing AI for real-world applications?\n3. How does machine learning enhance AI capabilities?\n4. Discuss ethical considerations and limitations of AI technologies."
            },
            {
                "input": "The study examines the role of renewable energy in combating climate change, focusing on solar and wind energy. It highlights the benefits, challenges, and potential policy implications of adopting renewable energy sources.",
                "output": "Viva/Presentation Questions:\n1. How can solar and wind energy address global carbon emission challenges?\n2. What are the main obstacles to implementing renewable energy at scale?\n3. Discuss potential policy measures to support renewable energy adoption.\n4. What are the economic trade-offs of transitioning to renewable energy?"
            }
        ],
        "instruction": "Generate insightful, academically relevant questions based on the content, aimed at fostering critical thinking and discussion."
    },
    "grading": {
        "examples": [
            {
                "input": "This project report provides a detailed study of AI's applications in healthcare, focusing on diagnostic tools, predictive analytics, and patient care systems. The rubric includes technical accuracy, innovation, structure, and clarity.",
                "output": "Grading Summary:\n• Technical Accuracy: High - The report accurately explains AI applications and provides evidence-based insights.\n• Innovation: Moderate - While impactful, the ideas largely build on existing solutions.\n• Structure: Excellent - Well-organized sections with logical flow.\n• Clarity: High - Technical terms are defined and explained with precision."
            }
        ],
        "instruction": "Grade the document based on rubric criteria, providing specific feedback for each criterion in a concise, evaluative summary."
    }
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def generate_response(prompt, feature):
    try:
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
    content = None
    output = None
    feature = request.form.get("feature")

    if request.method == "POST":
        pasted_text = request.form.get("pasted_text")
        file = request.files.get("file")

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

        if content and feature in FEW_SHOT_PROMPTS:
            prompt = f"{content}"
            output = generate_response(prompt, feature)

    return render_template("index.html", content=content, output=output)

def check_local_plagiarism(file_path):
    try:
        current_content = ""
        if file_path.lower().endswith('.pdf'):
            current_content = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()

        if not current_content:
            return []

        results = []
        all_files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*"))
        
        for other_file in all_files:
            if other_file != file_path and not os.path.basename(other_file).startswith('temp_'): 
                try:
                    other_content = ""
                    if other_file.lower().endswith('.pdf'):
                        other_content = extract_text_from_pdf(other_file)
                    else:
                        with open(other_file, 'r', encoding='utf-8') as f:
                            other_content = f.read()
                    
                    if other_content: 
                        similarity = difflib.SequenceMatcher(None, current_content, other_content).ratio()
                        
                        if similarity > 0.1: 
                            matched_blocks = get_matching_blocks(current_content, other_content)
                            if matched_blocks: 
                                results.append({
                                    'file': os.path.basename(other_file),
                                    'similarity': round(similarity * 100, 2),
                                    'matched_content': matched_blocks
                                })
                except Exception as e:
                    print(f"Error processing file {other_file}: {e}")
                    continue

        return results

    except Exception as e:
        print(f"Error in check_local_plagiarism: {e}")
        return []

def check_online_plagiarism(text):

    try:
        headers = {
            "Authorization": "riMUHI2kElitOU5DHPW0yxBIznkVQfd9oPMq3IFR9a3bcc8b",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "language": "en",
            "country": "us"
        }
        
        response = requests.post(WINSTON_API_URL, json=payload, headers=headers)
        response_data = response.json()
        
        score = response_data.get("result", {}).get("score", None)
        
        if score is not None:
            print(f"Plagiarism Score: {score}%")
            return score
        else:
            print("Unable to retrieve score from the API response.")
            return None
    except Exception as e:
        print(f"Error in check_online_plagiarism: {e}")
        return {"error": str(e)}


def get_matching_blocks(text1, text2, context_length=50):

    matcher = difflib.SequenceMatcher(None, text1, text2)
    matches = []
    
    for block in matcher.get_matching_blocks():
        if block.size > 20: 
            start = max(0, block.a - context_length)
            end = min(len(text1), block.a + block.size + context_length)
            matches.append(text1[start:end])
    
    return matches[:3] 

@app.route("/check_plagiarism", methods=["POST"])
def check_plagiarism():
    try:
        check_type = request.form.get("check_type")
        content = None
        file_path = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file and (allowed_file(file.filename) or file.filename == 'pasted_text.txt'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                
                if file.filename.lower().endswith('.pdf'):
                    content = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
        else:
            content = request.form.get("content")
            if content and check_type == "local":

                temp_filename = f"temp_{int(time.time())}.txt"
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        if check_type == "local" and file_path:
            results = check_local_plagiarism(file_path)
            if file_path.startswith(os.path.join(app.config["UPLOAD_FOLDER"], "temp_")):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
            return jsonify({"results": results})
        
        if check_type == "online" and content:
            results = check_online_plagiarism(content)
            return jsonify(results)
        
        return jsonify({"error": "Invalid request"})
    
    except Exception as e:
        print(f"Error in check_plagiarism: {e}") 
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True)