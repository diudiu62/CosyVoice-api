import argparse
import asyncio
import io
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
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
    allow_origins=["*"],  # 允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许的HTTP头部
)

# 读取模组路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

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


# 音频生成函数（流式输出）
async def generate_audio_stream(request: AudioRequest):
    set_all_random_seed(request.seed)
    prompt_speech_16k = load_wav(request.prompt_voice, 16000)

    # 根据模式选择推理方法
    if request.mode == 'zero_shot':
        result = await asyncio.to_thread(cosyvoice.inference_zero_shot, request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'instruct':
        result = await asyncio.to_thread(cosyvoice.inference_instruct2, request.tts_text, request.instruct_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'sft':
        result = await asyncio.to_thread(cosyvoice.inference_sft, request.tts_text, request.sft_dropdown, stream=request.stream, speed=request.speed)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    # 流式输出
    for i in result:
        audio_data = i['tts_speech'].numpy().flatten()
        audio_bytes = (audio_data * (2**15)).astype(np.int16).tobytes()
        yield audio_bytes

# 音频生成函数（非流式输出）
async def generate_audio_buffer(request: AudioRequest):
    set_all_random_seed(request.seed)
    prompt_speech_16k = load_wav(request.prompt_voice, 16000)
    
    # 根据模式选择推理方法
    if request.mode == 'zero_shot':
        result = await asyncio.to_thread(cosyvoice.inference_zero_shot, request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'instruct':
        result = await asyncio.to_thread(cosyvoice.inference_instruct2, request.tts_text, request.instruct_text, prompt_speech_16k, stream=request.stream, speed=request.speed)
    elif request.mode == 'sft':
        result = await asyncio.to_thread(cosyvoice.inference_sft, request.tts_text, request.sft_dropdown, stream=request.stream, speed=request.speed)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    # 非流式输出
    buffer = io.BytesIO()
    tts_speeches = [j['tts_speech'] for i, j in enumerate(result)]
    audio_data = torch.concat(tts_speeches, dim=1)
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


# 获音色列表
@app.get("/sft_spk")
async def get_sft_spk():
    sft_spk = cosyvoice.list_available_spks()  # 获取音色列表
    return JSONResponse(content=sft_spk)  # 返回 JSON 格式的响应




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    # 加载模型
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=True, fp16=False) 
    print("默认音色",cosyvoice.list_avaliable_spks())
    app.run(host='0.0.0.0', port=args.port)