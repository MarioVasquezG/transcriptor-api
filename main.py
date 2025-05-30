from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import os
import ffmpeg
import torch
from datetime import timedelta
from pyannote.audio import Pipeline
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Cargar modelo de Whisper una sola vez
whisper_model = whisper.load_model("medium")

# Configurar pipeline de Pyannote
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGINGFACE_TOKEN)


@app.get("/")
def home():
    return {"message": "API de transcripción activa"}

@app.post("/transcribir")
async def transcribir(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporal
        with NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
            temp_audio.write(await file.read())
            audio_path = temp_audio.name

        base = os.path.splitext(audio_path)[0]

        # Transcripción con Whisper
        result = whisper_model.transcribe(audio_path, language='es')
        texto_completo = "\n".join([seg["text"].strip() for seg in result["segments"]])

        # Guardar texto sin hablantes
        with open(f"{base}.txt", "w") as f:
            f.write(texto_completo)

        # Convertir audio a mono y 16kHz para Pyannote
        wav_path = f"{base}_opt.wav"
        ffmpeg.input(audio_path).output(wav_path, ar=16000, ac=1).overwrite_output().run(quiet=True)

        # Diarización
        diarization = pipeline(wav_path)
        segmentos = []
        for turno, _, speaker in diarization.itertracks(yield_label=True):
            start, end = turno.start, turno.end
            texto = ""
            for seg in result["segments"]:
                if seg["start"] >= start and seg["end"] <= end:
                    texto += seg["text"].strip() + " "
            if texto.strip():
                segmentos.append(f"{speaker} ({str(timedelta(seconds=int(start)))})\n{texto.strip()}\n")

        # Guardar texto con hablantes
        with open(f"{base}_hablantes.txt", "w") as f:
            f.write("\n".join(segmentos))

        return JSONResponse({
            "mensaje": "Transcripción completada",
            "archivo_sin_hablantes": os.path.basename(f"{base}.txt"),
            "archivo_con_hablantes": os.path.basename(f"{base}_hablantes.txt")
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
