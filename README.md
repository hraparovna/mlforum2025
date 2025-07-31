# mlforum2025
title: MLForum 2025
emoji: 🏆           # любая emoji или пусто
colorFrom: indigo
colorTo: cyan
sdk: gradio        # gradio | streamlit | static | docker
sdk_version: "4.31.1"   # опционально; уберите, если не важно
app_file: app.py   # точное имя вашего основного скрипта
pinned: false

This repository contains a Gradio application that runs a single-elimination tournament where an LLM judges pairs of agent forecasts.

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with your API key:
   ```bash
   API_KEY=YOUR_API_KEY_HERE
   ```

## Usage

Run the application with:
```bash
python app.py [--subset N]
```
The interface will display the logo and allow you to start the tournament. Answers are read from the `answers` folder and agent metadata from `agents.csv`.