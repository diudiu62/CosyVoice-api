import argparse
import asyncio
import io
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
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

# 预定义变量
max_val = 0.8

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


async def generate_audio(request: AudioRequest):
    '''音频生成函数'''
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

async def generate_audio_stream(request: AudioRequest):
    '''流式处理，返回音频数据流'''
    result = await generate_audio(request)
    for i in result:
        audio_data = i['tts_speech'].numpy().flatten()
        audio_bytes = (audio_data * (2**15)).astype(np.int16).tobytes()
        yield audio_bytes

async def generate_audio_buffer(request: AudioRequest):
    '''非流式处理，返回音频数据流'''
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
    

@app.post("/upload_prompt_audio")
async def upload_prompt_audio(file: UploadFile = File(...)):
    '''上传用于克隆的音频文件'''
    if not file.filename.endswith(('.wav', '.WAV', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .wav or .mp3 files are accepted.")

    # 读取上传的音频文件
    audio_data = await file.read()

    # 将其保存到本地
    output_path = os.path.join("audio_templates", file.filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查文件是否已经存在
    if os.path.exists(output_path):
        return {"filename": file.filename, "message": "Audio file already exists", "path": output_path}
    
    # 保存文件
    with open(output_path, "wb") as f:
        f.write(audio_data)

    return {"filename": file.filename, "message": "Audio file uploaded successfully", "path": output_path}

@app.get("/audio_templates")
async def get_audio_templates():
    '''获取audio_templates文件夹下的音频文件列表'''
    audio_folder = "audio_templates"
    try:
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.WAV', '.mp3'))]  # 只筛选.wav和.mp3格式的文件
        return JSONResponse(content={"status": "success", "data": {"audio_files": audio_files}})
    except FileNotFoundError:
        return JSONResponse(content={"status": "error", "message": "Audio templates folder not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.get("/sft_spk")
async def get_sft_spk():
    '''获取系统音色列表'''
    sft_spk = cosyvoice.list_available_spks()
    return JSONResponse(content=sft_spk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='local path or modelscope repo id')
    args = parser.parse_args()

    # 初始化CosyVoice模型
    cosyvoice = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, fp16=False)
    uvicorn.run(app, host='0.0.0.0', port=50000)
