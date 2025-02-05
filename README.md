# CosyVoice API 使用说明文档

## 概述

CosyVoice API 是一个用于文本转语音（TTS）的接口，允许用户通过指定文本、模式和其他参数生成语音内容。该 API 支持流式和非流式输出方式，并提供可用音色的列表。
## 准备
cosyvoice-api.py文件放到cosyvoice项目根目录，切换cosyvoice的依赖环境。
、、、
pip install fastapi
、、、

## 启动
、、、
python cosyvoice-api.py --port 50000 --model_dir path/to/your/model（模型路径）
、、、

## API 接口

### 1. 文本转语音接口

**请求方法：** `POST`  
**请求地址：** `/text-tts`

#### 请求参数

| 参数名         | 类型     | 必填   | 描述                                                     |
|--------------|--------|------|--------------------------------------------------------|
| tts_text     | string | 是    | 需要转换为语音的文本内容。                                    |
| mode         | string | 是    | 指定转换模式，可选值包括：`zero_shot`, `instruct`, `sft`。 |
| sft_dropdown | string | 否    | 自定义音色名称，仅在使用 `sft` 模式时需要。                 |
| prompt_text  | string | 否    | 额外的提示文本，仅在使用 `zero_shot` 模式时需要。          |
| instruct_text| string | 否    | 指令文本，仅在使用 `instruct` 模式时需要。                 |
| seed         | int    | 否    | 随机种子，用于控制生成的随机性，默认为 0。                    |
| stream       | bool   | 否    | 是否启用流式输出，默认为 false。                            |
| speed        | float  | 否    | 语音速度，默认为 1.0。                                     |
| prompt_voice | string | 否    | 提示音频文件的路径，默认为 `None`。                        |

#### 示例请求

```json
{
    "tts_text": "你好，欢迎使用 CosyVoice API。",
    "mode": "zero_shot",
    "seed": 42,
    "stream": true,
    "prompt_voice": "path/to/prompt.wav",
    "prompt_text": "（克隆音频的文案）"
}
```

#### 响应

- **成功响应**（流式输出）：
  - 内容类型：`audio/pcm`
  - 返回流式生成的音频数据。

- **成功响应**（非流式输出）：
  - 内容类型：`audio/wav`
  - 返回完整音频文件。

- **错误响应**：
  - 状态码 `400`：请求参数不正确。
  - 状态码 `500`：服务器生成音频失败。

### 2. 获取音色列表接口

**请求方法：** `GET`  
**请求地址：** `/sft_spk`

#### 响应

- **成功响应**：
  - 内容类型：`application/json`
  - 返回可用的音色列表。

```json
{
    "available_spks": ["voice_1", "voice_2", "voice_3"]
}