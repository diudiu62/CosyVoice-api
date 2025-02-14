import argparse
import asyncio
import io
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Optional
import numpy as np
import torch
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import torchaudio

# FastAPI实例
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# 读取模组路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

class AudioRequest(BaseModel):
    tts_text: str
    mode: str
    sft_dropdown: Optional[str] = None
    prompt_text: Optional[str] = None
    instruct_text: Optional[str] = None
    seed: Optional[int] = 0
    stream: Optional[bool] = False
    speed: Optional[float] = 1.0
    prompt_voice: Optional[str] = None

# 音频生成函数
async def generate_audio(request: AudioRequest):
    set_all_random_seed(request.seed)
    prompt_speech_16k = load_wav(request.prompt_voice, 16000) if request.prompt_voice else None

    inference_map = {
        'zero_shot': cosyvoice.inference_zero_shot,
        'instruct': cosyvoice.inference_instruct2,
        'sft': cosyvoice.inference_sft
    }

    if request.mode not in inference_map:
        raise HTTPException(status_code=400, detail="Invalid mode")

    args = None
    if request.mode == 'sft':
        args = (request.tts_text, request.sft_dropdown, request.stream, request.speed)
    elif request.mode == 'zero_shot':
        args = (request.tts_text, request.prompt_text, prompt_speech_16k, request.stream, request.speed)
    elif request.mode == 'instruct':
        args = (request.tts_text, request.instruct_text, prompt_speech_16k, request.stream, request.speed)
    
    try:
        result = await asyncio.to_thread(inference_map[request.mode], *args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation error: {str(e)}")

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    return result

# 流式处理
async def generate_audio_stream(request: AudioRequest):
    result = await generate_audio(request)
    for i in result:
        audio_data = i['tts_speech'].numpy().flatten()
        audio_bytes = (audio_data * (2**15)).astype(np.int16).tobytes()
        yield audio_bytes

# 非流式处理
async def generate_audio_buffer(request: AudioRequest):
    result = await generate_audio(request)
    buffer = io.BytesIO()
    audio_data = torch.cat([j['tts_speech'] for j in result], dim=1)
    torchaudio.save(buffer, audio_data, cosyvoice.sample_rate, format="wav")
    buffer.seek(0)
    return buffer

@app.post("/text-tts")
async def text_tts(request: AudioRequest):
    if not request.tts_text:
        raise HTTPException(status_code=400, detail="Query parameter 'tts_text' is required")
    
    if request.stream:
        # 流式输出
        return StreamingResponse(generate_audio_stream(request), media_type="audio/pcm")
    else:
        # 非流式输出
        buffer = await generate_audio_buffer(request)
        return Response(buffer.read(), media_type="audio/wav")

# 音色列表
@app.get("/sft_spk")
async def get_sft_spk():
    sft_spk = cosyvoice.list_available_spks()
    return JSONResponse(content=sft_spk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='local path or modelscope repo id')
    args = parser.parse_args()

    # 初始化CosyVoice模型
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=False)
    uvicorn.run(app, host='0.0.0.0', port=50000)