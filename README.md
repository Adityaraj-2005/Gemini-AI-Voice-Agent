
# Gemini AI Voice Agent

A Python-based voice assistant that listens to spoken questions,
uses Gemini to understand and answer them, and speaks responses
back to the user.

## Features

- Microphone-based voice recording
- Speech transcription using Gemini
- AI-generated answers
- Text-to-speech responses
- LangGraph workflow with conditional routing
- Stop command to end the conversation

## Tech Stack

- Python
- Google Gemini API
- LangGraph
- SoundDevice
- SciPy

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory and add:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own API key.

## Run

```bash
python main.py
```

Speak into your microphone when prompted.

## Note

A working Gemini API key, internet connection, microphone,
and speaker are required. Model availability depends on
the Google AI API account and supported models.
