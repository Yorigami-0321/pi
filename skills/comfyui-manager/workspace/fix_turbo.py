import json

with open('D:/Tools/pi/skills/comfyui-manager/workspace/data/anima-txt2img-aesthetic-lora-miaomiao.json') as f:
    wf = json.load(f)

# 添加 turbo LoRA，路径使用 Windows 反斜杠
# 在 Python 字符串中: 'Anima\\anima...' -> Anima\anima...
wf['64'] = {
    'class_type': 'LoraLoaderModelOnly',
    'inputs': {
        'model': ['44', 0],
        'lora_name': 'Anima\\anima-turbo-lora-v0.2.safetensors',
        'strength_model': 1.0
    }
}

wf['61']['inputs']['model'] = ['64', 0]
del wf['62']
del wf['63']

out_path = 'D:/Tools/pi/skills/comfyui-manager/workspace/data/anima-txt2img-aesthetic-lora-miaomiao-turbo-only.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2, ensure_ascii=False)

# 验证
with open(out_path) as f:
    wf2 = json.load(f)
val = wf2['64']['inputs']['lora_name']
print(f'lora_name value: {repr(val)}')
print(f'Expected: Anima\\anima-turbo-lora-v0.2.safetensors')
print(f'Correct: {val == "Anima\\anima-turbo-lora-v0.2.safetensors"}')
