from flask import Flask, request, jsonify
from threading import Thread
import whisper
import yt_dlp
import os
import openai
from config import transcription_example, summary_example

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/')
def transcribir():
    url = request.args.get('url')
    if not url:
        return {'error': 'URL parameter missing'}, 400

    # Download the video using yt-dlp
    temp_file_path = os.path.join('/tmp', 'audio.mp3')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)

            t1 = Thread(target=ydl.download, args=([url],))
            t1.start()

            # Wait for the download to complete
            t1.join()
    except Exception as e:
        return {'error': str(e)}, 400

    # Transcribe the audio using Whisper
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(temp_file_path)
        transcript = result["text"]
    except Exception as e:
        return {'error': str(e)}, 500
    finally:
        os.remove(temp_file_path)

    # Generate a summary using OpenAI
    try:
        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful program that summarizes any transcription."},
                {"role": "user", "content": transcription_example},
                {"role": "assistant", "content": summary_example},
                {"role": "user", "content": transcript}
            ]
        )
    except Exception as e:
        return {'error': str(e)}, 500
    
    # Return a JSON response
    response = {
        'title': video_title,
        'transcription': transcript,
        'summary': summary.choices[0].message
    }
    return jsonify(response)
