from flask import Flask, render_template, request, jsonify
import json
import subprocess
from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import openai
import requests
from googleapiclient.discovery import build
from flask_cors import CORS
import cv2 
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamPredictor

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
app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'app2/static/uploads'
MODEL_PATH = 'app2/sam_vit_b_01ec64.pth'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam.to(device)
predictor = SamPredictor(sam)

@app.route('/separated_features')
def separated_features():
    return render_template('index.html')  # Your separated features page

    

@app.route('/transition_page')
def transition_page():
    return render_template('transition.html', show_navbar=True)

@app.route('/process', methods=['POST'])
def process():
    # Handle uploaded files for image-based transitions
    before_file = request.files['before']
    after_file = request.files['after']
    transition_type = request.form['transition']

    # Save files
    before_path = os.path.join(UPLOAD_FOLDER, 'before.jpg')
    after_path = os.path.join(UPLOAD_FOLDER, 'after.jpg')
    before_file.save(before_path)
    after_file.save(after_path)

    # Process transition
    result_path = os.path.join(UPLOAD_FOLDER, 'result.mp4')
    if transition_type == 'fade':
        create_fade_transition(before_path, after_path, result_path)
    elif transition_type == 'slide':
        create_slide_transition(before_path, after_path, result_path)
    elif transition_type == 'jump_in':
        return jsonify({'error': 'Invalid transition type for images, use video inputs for Jump In.'})
    elif transition_type == 'blur':
        create_blur_transition(before_path, after_path, result_path)
    elif transition_type == 'rotate':   
        create_rotate_transition(before_path, after_path, result_path)

    # Return the result path
    return jsonify({'result': f'/static/uploads/{os.path.basename(result_path)}'}) # Ensure this path is correct for video playback

@app.route('/process_video', methods=['POST'])
def process_video():
    # Handle uploaded files for video-based transitions
    before_file = request.files['before']
    after_file = request.files['after']
    transition_type = request.form['transition']

    # Save files
    before_path = os.path.join(UPLOAD_FOLDER, 'before_video.mp4')
    after_path = os.path.join(UPLOAD_FOLDER, 'after_video.mp4')
    before_file.save(before_path)
    after_file.save(after_path)

    result_path = os.path.join(UPLOAD_FOLDER, 'result_jump_in.mp4')

    if transition_type == 'jump_in':
        create_jump_in_transition_video(before_path, after_path, result_path)
    else:
        return jsonify({'error': 'Invalid transition type for video, use "jump_in" for video transitions.'})

    # Return the result path
    return jsonify({'result': f'/static/uploads/result_jump_in.mp4'})  # Ensure this path is correct for video playback

# Basic transitions
def create_fade_transition(before_path, after_path, result_path):
    try:
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            raise ValueError("Error: Unable to read uploaded images.")

        height, width = before_img.shape[:2]
        after_img = cv2.resize(after_img, (width, height))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(result_path, fourcc, 15, (width, height))

        for alpha in np.linspace(0, 1, 15):
            blended = cv2.addWeighted(before_img, 1 - alpha, after_img, alpha, 0)
            out.write(blended)

        out.release()
        print(f"Transition video created at {result_path}")

    except Exception as e:
        print(f"Error during fade transition: {e}")
        
def create_slide_transition(before_path, after_path, result_path):
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)

    if before_img is None or after_img is None:
        print("Error: Unable to read uploaded images.")
        return

    height, width = before_img.shape[:2]
    after_img = cv2.resize(after_img, (width, height))

    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (width, height))

    # Create the sliding frames
    for offset in range(0, width, 20):  # Step size 20 for faster transition
        slide_frame = np.zeros_like(before_img)
        slide_frame[:, :width - offset] = before_img[:, offset:]
        slide_frame[:, width - offset:] = after_img[:, :offset]
        out.write(slide_frame)

    # Ensure the final frame is fully the "after" image
    out.write(after_img)
    out.release()


#Blur Transition
def create_blur_transition(before_path, after_path, result_path):
    try:
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            raise ValueError("Error: Unable to read uploaded images.")

        height, width = before_img.shape[:2]
        after_img = cv2.resize(after_img, (width, height))

        out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (width, height))

        # Blur the "before" image
        for k in range(1, 31, 2):  # Gradual increase in blur
            blurred_before = cv2.GaussianBlur(before_img, (k, k), 0)
            out.write(blurred_before)

        # Transition to the sharp "after" image
        for _ in range(10):  # Maintain sharp "after" image for a few frames
            out.write(after_img)

        out.release()
        print(f"Blur transition video created at {result_path}")
    except Exception as e:
        print(f"Error during blur transition: {e}")


#Rotate Transition 
def create_rotate_transition(before_path, after_path, result_path):
    try:
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            raise ValueError("Error: Unable to read uploaded images.")

        height, width = before_img.shape[:2]
        after_img = cv2.resize(after_img, (width, height))

        out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (width, height))

        center = (width // 2, height // 2)

        # Rotate the "before" image
        for angle in range(0, 360, 10):  # Faster rotation with 10° steps
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(before_img, M, (width, height))
            out.write(rotated)

        # Write the final frame as the complete "after" image
        out.write(after_img)
        out.release()
        print(f"Rotate transition video created at {result_path}")
    except Exception as e:
        print(f"Error during rotate transition: {e}")



# Advanced transitions
import mediapipe as mp

def create_jump_in_transition_video(before_path, after_path, result_path):
    # Open video files
    before_video = cv2.VideoCapture(before_path)
    after_video = cv2.VideoCapture(after_path)

    # Get video properties
    fps = int(before_video.get(cv2.CAP_PROP_FPS))
    width = int(before_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(before_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose   
    pose = mp_pose.Pose()

    # Function to track body pose keypoints
    def track_body_pose(video):
        frames = []
        keypoints_list = []

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Convert the frame to RGB (MediaPipe expects RGB format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose model
            result = pose.process(rgb_frame)

            # Extract the body keypoints if available
            if result.pose_landmarks:
                keypoints = [(landmark.x, landmark.y) for landmark in result.pose_landmarks.landmark]
                keypoints_list.append(keypoints)
            else:
                keypoints_list.append(None)

            frames.append(frame)

        return frames, keypoints_list

    # Function to detect the apex of the jump
    def find_apex(keypoints_list):
        # Track the y-coordinate of keypoints like feet or hips to determine the apex
        y_positions = []
        for keypoints in keypoints_list:
            if keypoints:  # Ensure keypoints exist for the frame
                # Using the y-coordinate of the left foot (keypoint 27 in MediaPipe)
                y_positions.append(keypoints[27][1] if len(keypoints) > 27 else None)

        # Find the frame with the lowest y-coordinate (highest point in the jump)
        apex_index = np.argmin(y_positions)
        return apex_index

    # Track the body poses for both clips
    before_frames, before_keypoints = track_body_pose(before_video)
    after_frames, after_keypoints = track_body_pose(after_video)

    # Find the apex for both clips
    before_apex_index = find_apex(before_keypoints)
    after_apex_index = find_apex(after_keypoints)

    print(f"Before Clip Apex Frame: {before_apex_index}")
    print(f"After Clip Apex Frame: {after_apex_index}")

    # Calculate transition duration (based on a percentage of the shorter clip)
    transition_duration = min(len(before_frames), len(after_frames)) // 5
    if transition_duration == 0:  # Ensure transition duration is at least 1 frame
        transition_duration = 1

    # Start writing the first part of the "before" clip (up to the apex frame)
    for i in range(before_apex_index + 1):
        out.write(before_frames[i])

    # Apply the transition by blending frames between the two clips
    transition_start = min(before_apex_index, after_apex_index)
    transition_end = transition_start + transition_duration

    for i in range(transition_start, transition_end):
        alpha = (i - transition_start) / transition_duration
        if before_apex_index + i < len(before_frames) and after_apex_index + i < len(after_frames):
            blended_frame = cv2.addWeighted(before_frames[before_apex_index + i], 1 - alpha,
                                            after_frames[after_apex_index + i], alpha, 0)
            out.write(blended_frame)

    # Write the remaining part of the longer clip
    if len(before_frames) > len(after_frames):
        for i in range(after_apex_index + transition_duration, len(before_frames)):
            out.write(before_frames[i])
    else:
        for i in range(after_apex_index + transition_duration, len(after_frames)):
            out.write(after_frames[i])

    # Release resources
    before_video.release()
    after_video.release()
    out.release()


# --- Database Setup ---
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, 
    username TEXT UNIQUE,
    password TEXT, 
    channel_id TEXT
)''')

# Insert dummy users
cursor.executemany('''INSERT OR IGNORE INTO users (username, password, channel_id) VALUES (?, ?, ?)''', [
    ('MrBeast', 'pass1', 'UCX6OQ3DkcsbYNE6H8uQQuVA'),
    ('ApnaCollege', 'pass2', 'UCBwmMxybNva6P_5VmxjzwqA'),
    ('MarquesBrownlee', 'pass3', 'UCBJycsmduvYEL83R_U4JriQ'),
    ('AbdulBari', 'pass4', 'UCZCFT11CWBi3MHNlGf019nw')
])
conn.commit()

# --- Database Setup for Caching ---
cursor.execute('''CREATE TABLE IF NOT EXISTS channel_videos (
    video_id TEXT PRIMARY KEY,
    title TEXT,
    published_at TEXT,
    views INTEGER,
    likes INTEGER,
    channel_id TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS global_trending_videos (
    video_id TEXT PRIMARY KEY,
    hour INTEGER,
    day INTEGER,
    views INTEGER,
    likes INTEGER
)''')
conn.commit()


# --- Helper Functions ---
def authenticate_user(username, password):
    cursor.execute("SELECT channel_id FROM users WHERE username = ? AND password = ?", (username, password))
    result = cursor.fetchone()
    return result[0] if result else None

# Channel Specific (based on historic data) -> best upload time
def get_channel_videos(channel_id, max_results=500):
    # Check if data is already cached
    cursor.execute("SELECT * FROM channel_videos WHERE channel_id = ?", (channel_id,))
    cached_data = cursor.fetchall()
    if cached_data:
        return pd.DataFrame(cached_data, columns=["Video ID", "Title", "Published At", "Views", "Likes", "Channel ID"])
    
    # Fetch data from API
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    videos = []
    next_page_token = None
    
    while len(videos) < max_results:
        request = youtube.search().list(
            part='snippet',
            channelId=channel_id,
            maxResults=50,
            order='date',
            type='video',
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            publish_time = item['snippet']['publishTime']
            metrics = get_video_metrics(video_id)
            videos.append({
                'Video ID': video_id,
                'Title': video_title,
                'Published At': publish_time,
                'Views': metrics['Views'],
                'Likes': metrics['Likes'],
                'Channel ID': channel_id
            })
            # Save to database
            cursor.execute('''
                INSERT OR IGNORE INTO channel_videos (video_id, title, published_at, views, likes, channel_id) 
                VALUES (?, ?, ?, ?, ?, ?)''', 
                (video_id, video_title, publish_time, metrics['Views'], metrics['Likes'], channel_id))
        conn.commit()
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break  # Exit if no more pages of results are available
    
    return pd.DataFrame(videos)

def get_video_metrics(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(part='statistics', id=video_id)
    response = request.execute()
    stats = response['items'][0]['statistics']
    return {'Views': int(stats.get('viewCount', 0)), 'Likes': int(stats.get('likeCount', 0))}


def preprocess_data(df):
    df['Published At'] = pd.to_datetime(df['Published At'])
    df['Hour'] = df['Published At'].dt.hour
    df['Day'] = df['Published At'].dt.dayofweek  # Monday=0, Sunday=6
    return df

def train_model(df, target_col):
    X = df[['Hour', 'Day']]
    y = df[target_col]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def recommend_best_time(model_views, model_likes):
    best_hour, best_day = None, None
    highest_prediction_views = 0
    highest_prediction_likes = 0
    feature_columns = ['Hour', 'Day']
    for hour in range(24):
        for day in range(7):
            pred_views = model_views.predict(pd.DataFrame([[hour, day]], columns=feature_columns))[0]
            pred_likes = model_likes.predict(pd.DataFrame([[hour, day]], columns=feature_columns))[0]
            if pred_views > highest_prediction_views:
                best_hour, best_day = hour, day
                highest_prediction_views = pred_views
                highest_prediction_likes = pred_likes
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return best_hour, day_names[best_day], highest_prediction_views, highest_prediction_likes

# 2 step model training to get predicted likes and views if user were to upload at global best time
def get_global_trending_videos(max_results=100):
    # Check if global trending data is cached
    cursor.execute("SELECT * FROM global_trending_videos")
    cached_data = cursor.fetchall()
    if cached_data:
        return pd.DataFrame(cached_data, columns=["Video ID", "Hour", "Day", "Views", "Likes"])
    
    # Fetch from API
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    videos = []
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        maxResults=max_results
    )
    response = request.execute()
    for video in response.get("items", []):
        snippet = video["snippet"]
        stats = video.get("statistics", {})
        publish_time = snippet["publishedAt"]
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0)) if "likeCount" in stats else 0
        hour = int(publish_time[11:13])  # Extract hour
        day = pd.to_datetime(publish_time).dayofweek  # Monday=0, Sunday=6
        videos.append({"Video ID": video['id'], "Hour": hour, "Day": day, "Views": views, "Likes": likes})
        # Save to database
        cursor.execute('''
            INSERT OR IGNORE INTO global_trending_videos (video_id, hour, day, views, likes)
            VALUES (?, ?, ?, ?, ?)''', 
            (video['id'], hour, day, views, likes))
    conn.commit()
    return pd.DataFrame(videos)


# Train RandomForest model for global best upload time predictions
def train_global_model(df):
    X = df[['Hour', 'Day']]
    y_views = df['Views']
    y_likes = df['Likes']
    
    global_model_views = RandomForestRegressor(n_estimators=100)
    global_model_likes = RandomForestRegressor(n_estimators=100)
    
    global_model_views.fit(X, y_views)
    global_model_likes.fit(X, y_likes)
    
    return global_model_views, global_model_likes

# Function to recommend the global best upload time
def recommend_best_upload_time_global(model_views, model_likes):
    best_hour, best_day = None, None
    highest_prediction_views = 0
    highest_prediction_likes = 0
    feature_columns = ['Hour', 'Day']
    
    for hour in range(24):
        for day in range(7):
            pred_views = model_views.predict(pd.DataFrame([[hour, day]], columns=feature_columns))[0]
            pred_likes = model_likes.predict(pd.DataFrame([[hour, day]], columns=feature_columns))[0]
            if pred_views > highest_prediction_views:
                best_hour, best_day = hour, day
                highest_prediction_views = pred_views
                highest_prediction_likes = pred_likes
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return best_hour, day_names[best_day], highest_prediction_views, highest_prediction_likes

# Collect channel-specific data (e.g., subscriber count, past engagement)
def collect_channel_data(channel_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.channels().list(
        part="statistics",
        id=channel_id
    )
    response = request.execute()
    
    if 'items' in response and len(response['items']) > 0:
        channel_data = response['items'][0]['statistics']
        
        subscriber_count = int(channel_data.get("subscriberCount", 0))
        view_count = int(channel_data.get("viewCount", 0))
        video_count = int(channel_data.get("videoCount", 0))
        
        return subscriber_count, view_count, video_count
    else:
        return 0, 0, 0

# Train model with both global and channel-specific data
# Function to train model with both global and channel-specific data
def train_model_with_global_and_channel_data(df, channel_id):
    # Get the top 50 trending videos
    global_data = get_global_trending_videos()

    # Train the global model with this data
    global_model_views, global_model_likes = train_global_model(global_data)

    global_best_hour, global_best_day, _, _ = recommend_best_upload_time_global(global_model_views, global_model_likes)
    
    # Collect the subscriber count data (this is channel-specific but we won't use it for prediction)
    subscriber_count, _, _ = collect_channel_data(channel_id)
    
    # Remove 'Subscriber Count' from features and train the model with only 'Hour' and 'Day'
    X = df[['Hour', 'Day']]  # Only use 'Hour' and 'Day'
    y_views = df['Views']
    y_likes = df['Likes']
    
    model_views = RandomForestRegressor(n_estimators=100)
    model_likes = RandomForestRegressor(n_estimators=100)
    
    # Train models
    model_views.fit(X, y_views)
    model_likes.fit(X, y_likes)
    
    return model_views, model_likes

def predict_for_channel_at_best_global_time(channel_id, global_model_views, global_model_likes):
    # Get the global best upload time (using integer values for 'Day' and 'Hour')
    global_best_hour, global_best_day, _, _ = recommend_best_upload_time_global(global_model_views, global_model_likes)
    
    # Collect the subscriber count data (this is channel-specific but we won't use it for prediction)
    subscriber_count, _, _ = collect_channel_data(channel_id)
    
    # Convert day name (e.g., 'Thursday') to integer (e.g., 3 for Thursday)
    day_name_to_int = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    # Convert the best day name to its corresponding integer
    global_best_day_int = day_name_to_int.get(global_best_day, -1)
    
    # Ensure 'global_best_day_int' is a valid integer (0 to 6)
    if global_best_day_int == -1:
        raise ValueError(f"Invalid day name: {global_best_day}")
    
    # Prepare input features with only 'Hour' and 'Day' (no 'Subscriber Count')
    X_input = pd.DataFrame([[global_best_hour, global_best_day_int]], columns=['Hour', 'Day'])
    
    # Ensure that the columns are in the same order and types as during training
    X_input = X_input[['Hour', 'Day']]  # Ensure correct order
    
    # Print training columns for verification
    print(f"Training feature columns: {X_input.columns}")
    print(f"Prediction features (X_input): {X_input.head()}")

    # Predict using the trained models (ensure the same columns are used)
    predicted_views = global_model_views.predict(X_input)[0]
    predicted_likes = global_model_likes.predict(X_input)[0]
    
    return predicted_views, predicted_likes

# --- Routes ---

# Home Page
@app.route('/')
def home():
    return render_template('index.html', show_navbar=False)

# Generate Script Page
@app.route('/generate-script-page')
def generate_script_page():
    return render_template('generate_script.html', show_navbar=True)

@app.route('/generate-script', methods=['POST'])
def generate_script():
    data = request.json
    video_idea = data.get('videoIdea', '')
    product = data.get('product', '')

    if not video_idea or not product:
        return jsonify({"error": "Both 'videoIdea' and 'product' are required!"}), 400

    persona = {
        "role": "system",
        "content": "You are a helpful assistant skilled in creating YouTube video scripts."
    }

    user_message = {
        "role": "user",
        "content": f"Generate a creative and engaging YouTube video script based on this idea: '{video_idea}'. Integrate '{product}' into the script naturally as a product placement."
    }

    messages = [persona, user_message]

    try:
        # Call OpenAI ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        response = completion["choices"][0]["message"]["content"].strip()
        return jsonify({"script": response})

    except Exception as e:
        # Log the error and send a friendly message to the frontend
        print(f"Error generating script: {str(e)}")  # You can log this to a file for better debugging
        return jsonify({"error": "Failed to generate script. Please try again."}), 500


# Recommend Images Page
# Function to fetch photos from Pexels API
def get_photos(query):
    headers = {
        'Authorization': PEXELS_API_KEY
    }
    params = {
        'query': query,
        'per_page': 5,  # Fetch 5 photos
        'page': 1
    }
    response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch photos:", response.status_code, response.text)
        return {}

# Function to fetch gifs from Giphy API
def get_giphy(query):
    params = {
        'api_key': GIPHY_API_KEY,
        'q': query,
        'limit': 5,  # Fetch 5 gifs
        'offset': 0,
        'rating': 'G',
        'lang': 'en'
    }
    response = requests.get('https://api.giphy.com/v1/gifs/search', params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch gifs:", response.status_code, response.text)
        return {}
    
@app.route('/recommend-images-page')
def recommend_images_page():
    return render_template('recommend_images.html', show_navbar=True)

@app.route('/recommend-images', methods=['POST'])
def recommend_images():
    query = request.form['query']
    
    # Fetch photos and gifs using the helper functions
    photos = get_photos(query)
    gifs = get_giphy(query)
    
    # Extract URLs for photos and gifs
    photo_urls = [{'url': photo['src']['original'], 'type': 'image'} for photo in photos.get('photos', [])]
    gif_urls = [{'url': gif['images']['original']['url'], 'type': 'gif'} for gif in gifs.get('data', [])]
    
    # Combine photo and gif URLs
    combined_urls = photo_urls + gif_urls
    
    return jsonify(combined_urls)

# Create a YouTube API client
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Recommend Music Page
@app.route('/recommend-music-page')
def recommend_music_page():
    return render_template('recommend_music.html', show_navbar=True)

@app.route('/recommend-music', methods=['POST'])
def recommend_music():
    query = request.form['query']  # Get the search query from the user input

    try:
        # Search for videos with the given query related to background music
        response = youtube.search().list(
            q=query,  # Use the query passed from the frontend
            part="snippet",
            maxResults=5,  # Limit results to top 5
            type="video",
            videoCategoryId="10"  # Music category
        ).execute()

        # Prepare the list of video recommendations
        videos = [
            {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'video_link': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in response['items']
        ]

        # Return the list of music videos as JSON
        return jsonify({"music_videos": videos})

    except Exception as e:
        # Return an error if something goes wrong
        print(f"Error recommending music: {str(e)}")
        return jsonify({"error": "Failed to recommend music. Please try again later."}), 500

# Trending Thumbnails Page
@app.route('/trending-page')
def trending_page():
    return render_template('trending.html', show_navbar=True)

@app.route('/trending', methods=['POST'])
def trending():
    try:
        # Get the topic from the request body
        data = request.get_json()
        query = data.get('topic', '').strip()

        if not query:
            return jsonify({"error": "Topic is required!"}), 400

        # Use yt-dlp to search for trending videos related to the topic
        cmd = ["yt-dlp", f"ytsearch5:{query}", "--print-json", "--skip-download"]

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


# View Analytics Page
@app.route('/view-analytics-page')
def view_analytics_page():
    return render_template('view_analytics.html', show_navbar=True)

import logging

@app.route('/view-analytics', methods=['POST'])
def view_analytics():
    data = request.json
    username = data.get('channel_name')
    password = data.get('password')

    if not username or not password:
        return jsonify({'status': 'failure', 'message': 'Username and password are required.'}), 400

    # Authenticate user
    channel_id = authenticate_user(username, password)
    if not channel_id:
        return jsonify({'status': 'failure', 'message': 'Invalid username or password.'}), 401

    try:
        # Fetch video data for the authenticated user's channel
        channel_df = get_channel_videos(channel_id)

        if not channel_df.empty:
            # Preprocess and train models
            channel_df = preprocess_data(channel_df)

            # Train models
            model_views = train_model(channel_df, "Views")
            model_likes = train_model(channel_df, "Likes")

            best_hour, best_day, prediction_views, prediction_likes = recommend_best_time(model_views, model_likes)

            # Global Analysis
            global_df = get_global_trending_videos()
            if not global_df.empty:
                model_global_views, model_global_likes = train_global_model(global_df)
                global_best_hour, global_best_day, global_prediction_views, global_prediction_likes = recommend_best_upload_time_global(model_global_views, model_global_likes)
            else:
                global_best_hour, global_best_day, global_prediction_views, global_prediction_likes = None, None, None, None

            # Train model with both global and channel-specific data
            model_views, model_likes = train_model_with_global_and_channel_data(channel_df, channel_id)

            # Predict views and likes for the channel at the global best time
            predicted_views, predicted_likes = predict_for_channel_at_best_global_time(channel_id, model_global_views, model_global_likes)

            return jsonify({
                'status': 'success',
                'channel_best_time': {'day': best_day, 'hour': best_hour, 'views': int(prediction_views), 'likes': int(prediction_likes)},
                'global_best_time': {'day': global_best_day, 'hour': global_best_hour, 'views': int(global_prediction_views), 'likes': int(global_prediction_likes)},
                'predicted_at_global_best': {'views': int(predicted_views), 'likes': int(predicted_likes)}
            })
        else:
            return jsonify({'status': 'failure', 'message': 'No data available for the selected channel.'}), 404

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'status': 'failure', 'message': f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Run Separated Features app on port 5002
