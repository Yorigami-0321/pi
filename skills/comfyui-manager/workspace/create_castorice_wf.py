import json, os

src = 'D:/Tools/pi/skills/comfyui-manager/workspace/data/anima-txt2img-aesthetic-lora.json'
with open(src, 'r', encoding='utf-8') as f:
    wf = json.load(f)

# 修改 AnimaBoosterLoader 的模型名
for nid, node in wf.items():
    if node['class_type'] == 'AnimaBoosterLoader':
        node['inputs']['model_name'] = 'miaomiaoRealskin_anima10.safetensors'
        print(f'Changed model to: miaomiaoRealskin_anima10.safetensors')
        break

out = 'D:/Tools/pi/skills/comfyui-manager/workspace/data/anima-txt2img-aesthetic-lora-miaomiaoRealskin.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(wf, f, indent=2, ensure_ascii=False)

# 检查所有连接 ID 是否为字符串（ComfyUI API 要求）
errors = []
for nid, node in wf.items():
    for iname, val in node.get('inputs', {}).items():
        if isinstance(val, list) and len(val) == 2:
            if not isinstance(val[0], str):
                errors.append(f'  Node {nid}.{iname}: ID {val[0]} is {type(val[0]).__name__}, should be str')

if errors:
    print('ERRORS:')
    for e in errors:
        print(e)
else:
    print('All connection IDs are strings - OK')

print(f'Saved to {out}')
