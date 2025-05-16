import openai
import json
import subprocess
from flask import Flask, request, jsonify, render_template
import requests
from googleapiclient.discovery import build
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment

# Access the keys
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")
giphy_api_key = os.getenv("GIPHY_API_KEY")

import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment

# Access the keys
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")
giphy_api_key = os.getenv("GIPHY_API_KEY")


openai.api_key = openai_api_key


# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/end_to_end')
def end_to_end():
    return render_template('index.html')  # Your end-to-end solution page

# Helper function to call OpenAI API
def generate_script(video_idea, product):
    try:
        persona = {
            "role": "system",
            "content": "You are a helpful assistant skilled in creating YouTube video scripts based on video idea and the product that needs placement."
        }

        user_message = {
            "role": "user",
            "content": f"Generate a creative and engaging YouTube video script based on this idea: '{video_idea}'. Integrate '{product}' into the script naturally as a product placement."
        }

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[persona, user_message]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating script: {str(e)}")
        return None

# Helper function to analyze the script and extract a genre/keyword
def extract_genre_or_keyword(script):
    try:
        persona = {
            "role": "system",
            "content": "You are a helpful assistant skilled in determining suitable background music genres based on video script."
        }

        user_message = {
            "role": "user",
            "content": f"Suggest a suitable genre or keyword for background music for a video about: '{script}' with one word or few words only."
        }

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[persona, user_message]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error determining music genre: {str(e)}")
        return None

# Helper function to fetch background music suggestions using YouTube API
def fetch_music_suggestions(keyword):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.search().list(
            q=keyword,
            part="snippet",
            maxResults=5,
            type="video",
            videoCategoryId="10"  # Music category
        ).execute()

        return [
            {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'video_link': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in response.get('items', [])
        ]
    except Exception as e:
        print(f"Error fetching music suggestions: {str(e)}")
        return []

# Helper function to fetch photos from Pexels API
def get_photos(query):
    try:
        headers = {'Authorization': PEXELS_API_KEY}
        params = {'query': query, 'per_page': 5, 'page': 1}
        response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
        response.raise_for_status()
        return [photo['src']['original'] for photo in response.json().get('photos', [])]
    except Exception as e:
        print(f"Error fetching photos: {str(e)}")
        return []

# Helper function to fetch gifs from Giphy API
def get_gifs(query):
    try:
        params = {
            'api_key': GIPHY_API_KEY,
            'q': query,
            'limit': 5,
            'offset': 0,
            'rating': 'G',
            'lang': 'en'
        }
        response = requests.get('https://api.giphy.com/v1/gifs/search', params=params)
        response.raise_for_status()
        return [gif['images']['original']['url'] for gif in response.json().get('data', [])]
    except Exception as e:
        print(f"Error fetching gifs: {str(e)}")
        return []

# Helper function to fetch thumbnail suggestion
 
@app.route('/generate-script', methods=['POST'])
def generate_script_endpoint():
    data = request.json
    video_idea = data.get('videoIdea', '')
    product = data.get('product', '')

    if not video_idea or not product:
        return jsonify({"error": "Both 'videoIdea' and 'product' are required!"}), 400

    # Generate script
    script = generate_script(video_idea, product)
    if not script:
        return jsonify({"error": "Failed to generate script."}), 500
    
    # Extract genre or keyword for music based on the generated script
    keyword = extract_genre_or_keyword(script)
    if not keyword:
        return jsonify({"error": "Failed to extract music keyword."}), 500

    # Fetch music suggestions using the extracted keyword
    music_suggestions = fetch_music_suggestions(keyword)
    if not music_suggestions:
        return jsonify({"error": "Failed to fetch music suggestions."}), 500
    
    # Return the generated script along with music suggestions
    return jsonify({"script": script, "musicSuggestions": music_suggestions})

@app.route('/recommend-images', methods=['POST'])
def recommend_images():
    data = request.json
    video_idea = data.get('videoIdea', '')
    
    # Fetch photos and gifs using the helper functions
    photos = get_photos(video_idea)
    gifs = get_gifs(video_idea)
    
    # Extract URLs for photos and gifs
    photo_urls = [{'url': photo, 'type': 'image'} for photo in photos]
    gif_urls = [{'url': gif, 'type': 'gif'} for gif in gifs]
    
    # Combine photo and gif URLs
    combined_urls = photo_urls + gif_urls
    
    return jsonify(combined_urls)


@app.route('/trending', methods=['POST'])
def get_trending_youtube():
    try:
        # Get the topic from the request body
        data = request.json
        video_idea = data.get('videoIdea', '')

        # Use yt-dlp to search for trending videos related to the topic
        cmd = ["yt-dlp", f"ytsearch5:{video_idea}", "--print-json", "--skip-download"]

        # Run the yt-dlp command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)

        # Parse the output into JSON objects
        output = result.stdout.splitlines()
        videos = [json.loads(video) for video in output]

        # Extract only the first 5 thumbnail URLs
        thumbnails = [video.get("thumbnail") for video in videos[:5]]  # Limit to 5 thumbnails

        return jsonify({"thumbnails": thumbnails})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Command failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run End to End app on port 5001
