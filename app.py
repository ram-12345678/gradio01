from flask import Flask, request, jsonify, send_file
import gradio as gr
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import numpy as np
import io

app = Flask(__name__)

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
sample_rate = model.config.sampling_rate

# Text-to-Speech function
def text_to_speech(text, src_lang="eng", tgt_lang="arb"):
    text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    return sample_rate, audio_array_from_text

# Speech-to-Speech function
def speech_to_speech(audio, src_lang="eng", tgt_lang="rus"):
    audio, orig_freq = torchaudio.load(audio)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    return sample_rate, audio_array_from_audio

# Speech-to-Text function
def speech_to_text(audio, src_lang="eng", tgt_lang="ces"):
    audio, orig_freq = torchaudio.load(audio)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text_from_audio

# Text-to-Text function
def text_to_text(text, src_lang="eng", tgt_lang="ces"):
    text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text_from_text

@app.route('/')
def home():
    return "Welcome to the SeamlessM4T Translation Service"

@app.route('/text-to-speech', methods=['POST'])
def handle_text_to_speech():
    data = request.json
    text = data.get('text')
    src_lang = data.get('src_lang', 'eng')
    tgt_lang = data.get('tgt_lang', 'arb')
    sample_rate, audio = text_to_speech(text, src_lang, tgt_lang)
    audio_io = io.BytesIO()
    torchaudio.save(audio_io, torch.tensor(audio).unsqueeze(0), sample_rate)
    audio_io.seek(0)
    return send_file(audio_io, mimetype='audio/wav')

@app.route('/speech-to-speech', methods=['POST'])
def handle_speech_to_speech():
    audio_file = request.files['audio']
    src_lang = request.form.get('src_lang', 'eng')
    tgt_lang = request.form.get('tgt_lang', 'rus')
    audio_path = "temp.wav"
    audio_file.save(audio_path)
    sample_rate, audio = speech_to_speech(audio_path, src_lang, tgt_lang)
    audio_io = io.BytesIO()
    torchaudio.save(audio_io, torch.tensor(audio).unsqueeze(0), sample_rate)
    audio_io.seek(0)
    return send_file(audio_io, mimetype='audio/wav')

@app.route('/speech-to-text', methods=['POST'])
def handle_speech_to_text():
    audio_file = request.files['audio']
    src_lang = request.form.get('src_lang', 'eng')
    tgt_lang = request.form.get('tgt_lang', 'ces')
    audio_path = "temp.wav"
    audio_file.save(audio_path)
    text = speech_to_text(audio_path, src_lang, tgt_lang)
    return jsonify({'translated_text': text})

@app.route('/text-to-text', methods=['POST'])
def handle_text_to_text():
    data = request.json
    text = data.get('text')
    src_lang = data.get('src_lang', 'eng')
    tgt_lang = data.get('tgt_lang', 'ces')
    translated_text = text_to_text(text, src_lang, tgt_lang)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
