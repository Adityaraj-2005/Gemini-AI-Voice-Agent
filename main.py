import os
import base64
import sounddevice as sd

from scipy.io.wavfile import write, read
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ---------------- CONFIGURATION ----------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY nahi mili. Apni .env file check karo."
    )

client = genai.Client(api_key=API_KEY)

# Gemini models
CHAT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-lite-tts"

SAMPLE_RATE = 16000
DURATION = 5

STOP_WORDS = {
    "stop",
    "exit",
    "quit",
    "band karo",
    "बंद करो"
}


# ---------------- GRAPH STATE ----------------

class VoiceState(TypedDict):
    question: str
    answer: str


# ---------------- NODE 1: LISTEN ----------------

def listen(state: VoiceState):
    print("\n🎙️ Bolo (5 seconds)...")

    # Record audio from the microphone
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    # Save recorded audio
    write("question.wav", SAMPLE_RATE, recording)

    # Read audio file
    with open("question.wav", "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Gemini converts speech into text
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            (
                "Transcribe the speech in this audio. "
                "Keep the original language, Hindi or English. "
                "Return only the spoken words. "
                "Do not answer the question. "
                "If no speech is audible, return an empty response."
            ),
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav"
            )
        ]
    )

    question = (response.text or "").strip()

    print("Tum:", question)

    return {"question": question}


# ---------------- NODE 2: ROUTING ----------------

def route_question(state: VoiceState):
    question = " ".join(
        state["question"].lower().strip().split()
    ).strip(" .!?,")

    if question in STOP_WORDS:
        return "stop"

    if not question:
        return "empty"

    return "answer"


# ---------------- NODE 3: ANSWER ----------------

def answer_question(state: VoiceState):
    print("\n🤖 Gemini soch raha hai...")

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=(
            "You are a helpful AI voice assistant. "
            "Answer in simple Hindi. "
            "Use easy examples when helpful. "
            "Keep the response short enough to speak aloud. "
            "If the question is in English, you may still "
            "answer in simple Hindi.\n\n"
            "User's question: " + state["question"]
        )
    )

    answer = (response.text or "").strip()

    print("🤖 AI:", answer)

    return {"answer": answer}


# ---------------- NODE 4: SPEAK ----------------

def speak_answer(state: VoiceState):
    answer = state["answer"].strip()

    if not answer:
        print("Koi answer generate nahi hua.")
        return {}

    print("\n🔊 AI bol raha hai...")

    # Convert AI text into speech
    response = client.interactions.create(
        model=TTS_MODEL,

        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": (
                                    "Speak clearly and naturally "
                                    "in simple Hindi."
                                )
                            }
                        ]
                    }
                ]
            }
        ],

        response_format={"type": "audio"},

        generation_config={
            "speech_config": [
                {"voice": "Kore"}
            ]
        }
    )

    # Decode the audio returned by Gemini
    audio_bytes = base64.b64decode(
        response.output_audio.data
    )

    # Save generated audio
    with open("answer.wav", "wb") as audio_file:
        audio_file.write(audio_bytes)

    # Play audio on the laptop speaker
    sample_rate, audio_data = read("answer.wav")

    sd.play(audio_data, sample_rate)
    sd.wait()

    return {}


# ---------------- BUILD LANGGRAPH ----------------

builder = StateGraph(VoiceState)

builder.add_node("listen", listen)
builder.add_node("answer", answer_question)
builder.add_node("speak", speak_answer)

# Starting point
builder.add_edge(START, "listen")

# Check the question before answering
builder.add_conditional_edges(
    "listen",
    route_question,
    {
        "answer": "answer",
        "stop": END,
        "empty": END
    }
)

# Answer first, then speak
builder.add_edge("answer", "speak")
builder.add_edge("speak", END)

app = builder.compile()


# ---------------- MAIN LOOP ----------------

def main():
    print("=" * 40)
    print("       GEMINI AI VOICE AGENT")
    print(" Say 'stop' to end the conversation")
    print("=" * 40)

    while True:
        try:
            result = app.invoke({
                "question": "",
                "answer": ""
            })

            question = " ".join(
                result["question"].lower().strip().split()
            ).strip(" .!?,")

            # Exit without generating an AI answer
            if question in STOP_WORDS:
                print("\nVoice agent band ho gaya.")
                break

            if not question:
                print("Speech detect nahi hui. Dobara try karo.")

        except KeyboardInterrupt:
            print("\nVoice agent band ho gaya.")
            break

        except Exception as error:
            print("\nError:", error)
            print(
                "API key, internet, model access aur "
                "microphone check karo."
            )

            try:
                input("Retry karne ke liye Enter dabao...")
            except KeyboardInterrupt:
                print("\nVoice agent band ho gaya.")
                break


if __name__ == "__main__":
    main()
