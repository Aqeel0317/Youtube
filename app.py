import os
import json
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


# --- Configuration ---
load_dotenv()

# Flask App Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_change_me')

# Gemini API Configuration using google.generativeai
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    raise RuntimeError(f"Failed to configure or initialize Gemini API: {e}")

# --- Helper function to extract video ID from YouTube URL ---
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['youtu.be']:
        return parsed_url.path[1:]
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            qs = parse_qs(parsed_url.query)
            return qs.get('v', [None])[0]
        if parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        if parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    return None

# --- Helper function to fetch the YouTube video transcript ---
def get_transcript(video_id, lang_code='en'):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang_code])
        transcript = " ".join(item['text'] for item in transcript_list)
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# --- Helper function to analyze the transcript using Gemini API ---
def analyze_transcript(transcript):
    prompt = f"""
    Analyze the following YouTube video transcript and return your response as a strict JSON object with the following structure:

    {{
      "summary": "A concise summary of the video content.",
      "keypoints": ["Important point 1", "Important point 2", ...],
      "topics": ["Topic 1", "Topic 2", ...],
      "topic_explanations": {{
        "Topic 1": "A brief explanation of Topic 1 based on the transcript.",
        "Topic 2": "A brief explanation of Topic 2 based on the transcript.",
        ...
      }},
      "transcript": "The full transcript of the video."
    }}

    For each topic identified, provide a concise explanation of two to three lines based on the information present in the transcript.

    Transcript: {transcript}
    """
    try:
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip()
        print("Raw Gemini Response (Analysis):\n", raw_text)

        # Attempt more robust JSON parsing
        try:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the JSON string
                json_string = raw_text[start_index:end_index + 1]
                # Further refine to handle potential extra characters after the JSON
                try:
                    analysis = json.loads(json_string)
                    return analysis
                except json.JSONDecodeError as e_inner:
                    # Attempt to find the last valid JSON object by iteratively removing characters
                    temp_string = json_string
                    while temp_string:
                        try:
                            analysis = json.loads(temp_string)
                            return analysis
                        except json.JSONDecodeError:
                            temp_string = temp_string[:-1]
                    print(f"JSON Decode Error (Iterative Refinement): {e_inner}")
                    return {"error": f"The model's response was not valid JSON after refinement: {e_inner}"}
            else:
                print("Could not find valid JSON boundaries in the raw response.")
                return {"error": "The model's response did not contain valid JSON."}
        except json.JSONDecodeError as json_err:
            print(f"JSON Decode Error (Robust Parsing): {json_err}")
            return {"error": f"The model's response was not valid JSON: {json_err}"}

    except Exception as e:
        print(f"Gemini API error (Analysis): {e}")
        return {"error": str(e)}

# --- Helper function to answer questions using Gemini API ---
def answer_question(question, transcript):
    prompt = f"""
    Based on the following YouTube video transcript, answer the question asked by the user.
    Keep your answer concise and directly related to the content of the transcript.

    Transcript: {transcript}

    Question: {question}

    Answer:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error (Answering): {e}")
        return "Sorry, I could not generate an answer at this time."

# --- Flask routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        video_url = request.form.get("video_url")
        video_id = extract_video_id(video_url)
        if not video_id:
            return "Invalid YouTube URL provided.", 400

        transcript = get_transcript(video_id)  # Try English first
        if not transcript:
            transcript = get_transcript(video_id, lang_code='hi')  # Try Hindi if English fails

        if transcript:
            analysis = analyze_transcript(transcript)
            if analysis:
                return render_template("result.html", analysis=analysis)
            else:
                return "Error analyzing the transcript using Google Gemini API.", 500
        else:
            return "Unable to extract transcript from the video. It may be unavailable in the specified languages.", 500

    return render_template("index.html")

@app.route('/ask_question', methods=['POST'])
def ask_question_route():
    data = request.get_json()
    question = data.get('question')
    transcript = data.get('transcript')

    if not question or not transcript:
        return jsonify({"error": "Missing question or transcript."}), 400

    answer = answer_question(question, transcript)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)