

# 🎬 Team\_Untitled – Co-Create: The AI Studio for YouTubers 

🏅 **4th Place Winner** at **Synapse IEEE Winter Hackathon 2024**

**Co-Create** is an AI-powered platform tailored for **YouTubers**, helping them ideate, produce, and optimize video content — **end-to-end**. From intelligent product placement and scene scripting to transition automation and upload timing analytics, Co-Create redefines content creation with speed, precision, and engagement in mind.

> 💡 **Bridging AI with Creator Economy**

---

## 📚 Table of Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [How It Works](#3-how-it-works)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Contributors](#7-contributors)

---

## 1. Overview

The modern content creator needs **speed, relevance, and reach**. While platforms like **Canva, Descript, and Pictory** help with parts of the workflow, **Co-Create** offers a **one-stop solution** with a YouTube-first mindset.

Co-Create solves two core problems:

1. **Product Placement Done Right** – Creators often struggle to organically integrate sponsored products into their content. Co-Create generates **context-appropriate scripts, scenes, and placements** around the product with ease.
2. **Time-Intensive Transitions Simplified** – Advanced transitions like “jump cuts” require manual frame-by-frame editing. Co-Create automates it using computer vision — just upload before-and-after clips, and the AI handles the rest.

---

## 2. Features

### 🧠 End-to-End Video Ideation

* Just enter a **product** and an **idea**, and Co-Create:

  * Generates a full script
  * Suggests trending **visuals, gifs**
  * Provides **high-conversion thumbnail** templates
  * Recommends **background music**

### 🎯 AI-Driven Product Placement

* Contextual product mentions in script
* Scene ideas that feel organic
* Avoids awkward forced ads

### 🎞️ Transition Automation

* **Simple to advanced transitions** (e.g., fade-ins to “jump cut”)
* Upload two clips → get a **seamless edit**
* Saves hours of manual frame matching

### 📈 YouTube Analytics Integration

* Uses **YouTube Data API** to analyze:

  * Sentiment trends
  * Engagement rates
* Recommends **optimal upload timings**:

  * Global best practice
  * Personalized, based on user’s past schedule
* Predicts potential reach: **likes & views estimates**

---

## 3. How It Works

### 🧩 System Architecture

1. **Frontend (HTML, CSS)**

   * Option to go with End-To-End content creation by Co-Create or can choose to utilise Co-Create for specific parts of the content creation process.
   * Friendly UI inspired by popular creator tools

2. **Backend (Python + FastAPI)**

   * **OpenAI API**: Script + scene + thumbnail generation
   * **Pexels API**: Royalty-free media suggestions
   * **MediaPipe + OpenCV + Segment Anything**: For automated transitions
   * **YouTube Data API**: For analytics and predictions

3. **Content Generation Pipeline**

   * Product keyword → context-aware script & placement
   * Background music + image/video/GIF suggestions
   * Top thumbnail layouts from similar trending videos

4. **Transition Pipeline**

   * Before/after video → segment objects → align frames → insert transition (based on the choice made by the user given the different options provided)

5. **Analytics Engine**

   * Pulls channel statistics
   * Provides insights based on user's schedule as well as the global best practices
   * Predicts views/likes per timing slot

---

## 4. Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RISHASUN001/Synapse_Team-Untitled.git
cd Synapse_Team-Untitled
```

### 2. Set Up Environment Variables

Create a `.env` file with the following:

```bash
YOUTUBE_API_KEY = 'your_youtube_api_key'
openai.api_key = 'your_openai_api_key'
PEXELS_API_KEY = 'your_pexels_api_key'
GIPHY_API_KEY = 'your_giphy_api_key'
```

---

## 5. Usage

1. Launch the app via browser
2. Enter your video **idea** + **product**
3. Choose:

   * 🎬 End-to-end generation
   * ✍️ Just Script
   * 🌃 Just Thumbnails
   * 🌇 Just Gifs/Images
   * 🎷 Just Background Music
   * 📊 Just Youtube Analytics
   * 🎞️ Just Transition
     
4. Get:

   * Script + media assets
   * Thumbnail templates
   * Background music
   * Seamless video transitions
   * Upload time suggestions & engagement predictions

---


## 6. Contributors

* 👩‍💻 **Risha Sunil Shetty** – [@RISHASUN001](https://github.com/RISHASUN001)
* 👩‍💻 **Janhavee Singh** – [@JanhaveeSingh](https://github.com/JanhaveeSingh)
* 👩‍💻 **Yi Hsuen Cheng** – [@yiihsuenn](https://github.com/yiihsuenn)
* 👩‍💻 **Thwun Thiri Thu** – [@thiriii](https://github.com/thiriii)


