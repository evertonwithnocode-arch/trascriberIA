import runpod
from main import transcribe_video, YouTubeRequest
from fastapi.encoders import jsonable_encoder

async def handler(event):
    body = event.get("input", {})

    url = body.get("url")
    if not url:
        return {"error": "Nenhuma URL recebida."}

    req = YouTubeRequest(url=url)
    response = await transcribe_video(req)
    return jsonable_encoder(response)

runpod.serverless.start({"handler": handler})
