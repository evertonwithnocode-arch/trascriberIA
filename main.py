from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from typing import List
import os
import yt_dlp
import threading
import uuid
import time
import traceback
import re

from openai import OpenAI

# ======================================================
# 🔹 Inicialização OpenAI
# ======================================================
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ======================================================
# 🔹 Inicialização FastAPI e Tradutor
# ======================================================
app = FastAPI()
translator = GoogleTranslator(source='auto', target='pt')  # PT-BR

# ======================================================
# 🔥 Função de carregamento inteligente (GPU → CPU)
# ======================================================
def load_whisper_with_fallback():
    print("🔍 Verificando GPU CUDA...")
    try:
        model = WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16",
        )
        print("⚡ CUDA detectada — usando distil-large-v3!")
        return model
    except Exception:
        print("❌ CUDA indisponível. Alternando para CPU...")
        model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )
        print("🖥️ Usando CPU (tiny int8).")
        return model

print("🔄 Inicializando modelo Faster-Whisper...")
model = load_whisper_with_fallback()
print("✅ Modelo pronto.")

# ======================================================
# 🔄 CORS
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://transcribeia.lovable.app",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 📦 Modelos de request
# ======================================================
class YouTubeRequest(BaseModel):
    url: str

# ======================================================
# 🔢 Download + conversão WAV 16 kHz
# ======================================================
def download_audio(url: str):
    print("🎧 Baixando e convertendo áudio...")
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "outtmpl": "audio.%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title", "unknown")
    print(f"🎵 Áudio '{title}' baixado e convertido para WAV 16kHz.")
    return title, "audio.wav"

# ======================================================
# 🟡 Sistema de jobs
# ======================================================
jobs = {}  # job_id -> {status, title, transcription, summary, error}

# ======================================================
# ⚡ Função para dividir áudio em chunks
# ======================================================
def split_audio(file_path: str, chunk_length_ms: int = 25*60*1000) -> List[str]:
    print("✂️ Dividindo áudio em chunks...")
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_name = f"chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)
        print(f"   ✔ Chunk {chunk_name} criado.")
    return chunks

# ======================================================
# 🔧 Função de tradução segura em blocos
# ======================================================
def safe_translate(text: str) -> str:
    text = re.sub(r'\b(A\.I\.)\s+(?:\1\s+)+', r'\1 ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    translated_text = ""
    block = ""
    max_len = 4500
    
    for sentence in sentences:
        if len(block) + len(sentence) < max_len:
            block += " " + sentence
        else:
            try:
                translated_text += " " + translator.translate(block.strip())
            except Exception as e:
                print(f"⚠️ Erro traduzindo bloco: {e}")
                translated_text += " " + block.strip()
            block = sentence
    if block:
        try:
            translated_text += " " + translator.translate(block.strip())
        except Exception as e:
            print(f"⚠️ Erro traduzindo último bloco: {e}")
            translated_text += " " + block.strip()
    
    return translated_text.strip()

# ======================================================
# 🔧 Função para transcrever e traduzir chunk (com logs)
# ======================================================
def transcribe_and_translate_chunk(chunk_path: str, idx: int) -> str:
    try:
        print(f"🎙️ Iniciando transcrição do chunk {idx}...")
        segments, _ = model.transcribe(
            chunk_path,
            language=None,
            beam_size=1,
            vad_filter=False
        )
        text = "".join([seg.text for seg in segments])
        translated = safe_translate(text)
        print(f"✅ Chunk {idx} transcrito e traduzido.")
        return translated
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            print(f"🗑️ Chunk {idx} removido do disco.")

# ======================================================
# 🔹 Função de resumo usando OpenAI direto (com logs)
# ======================================================
def summarize_text(text: str) -> str:
    try:
        print("📝 Iniciando resumo do texto...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente especializado em resumir transcrições de reuniões de câmaras de vereadores. "
                        "Seu objetivo é gerar um resumo claro e objetivo destacando apenas os principais pontos discutidos, decisões tomadas, e ações definidas. "
                        "Evite repetições, comentários irrelevantes ou conversas paralelas. "
                        "O resumo deve ser organizado, de preferência em tópicos ou bullet points, e fácil de ler."
                        "e começar com o título 'Principais Pontos:'"
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        summary = response.choices[0].message.content.strip()
        print("✅ Resumo gerado com sucesso.")
        return summary
    except Exception as e:
        print(f"⚠️ Erro ao gerar resumo: {e}")
        return ""

# ======================================================
# 🔵 Endpoint iniciar transcrição
# ======================================================
@app.post("/start-transcription")
async def start_transcription(request: YouTubeRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "title": None, "transcription": None, "summary": None, "error": None}
    threading.Thread(target=process_job, args=(job_id, request.url)).start()
    print(f"📌 Job {job_id} criado e iniciado em thread.")
    return {"job_id": job_id}

# ======================================================
# 🔧 Função processa job (chunks + threads) com logs
# ======================================================
def process_job(job_id: str, url: str):
    try:
        print(f"🚀 JOB {job_id} iniciado para URL: {url}")
        start_time = time.time()
        title, audio_path = download_audio(url)
        jobs[job_id]["title"] = title

        chunks = split_audio(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("🗑️ Arquivo de áudio original removido após chunking.")

        print(f"🧩 {len(chunks)} chunks criados. Iniciando transcrição + tradução paralela...")

        results = ["" for _ in chunks]
        threads = []

        for i, chunk_path in enumerate(chunks):
            t = threading.Thread(target=lambda idx, path: results.__setitem__(idx, transcribe_and_translate_chunk(path, idx)), args=(i, chunk_path))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        full_text = " ".join(results)
        summary = summarize_text(full_text)
        duration = round(time.time() - start_time, 2)

        jobs[job_id].update({
            "status": "done",
            "summary": summary,
            "duration": duration
        })
        print(f"✅ JOB {job_id} finalizado em {duration}s")

    except Exception as e:
        print(f"❌ ERRO NO JOB {job_id}")
        print(traceback.format_exc())
        jobs[job_id].update({"status": "error", "error": str(e)})

# ======================================================
# 🔎 Endpoint status do job
# ======================================================
@app.get("/job-status/{job_id}")
async def transcription_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return jobs[job_id]
