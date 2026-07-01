import json, sys, os, math

# 节点类型到 input/output slot 定义
NODE_DEFS = {
    'CLIPLoader': {
        'inputs': [
            {'name': 'clip_name', 'type': 'COMBO', 'widget': {'name': 'clip_name'}},
            {'name': 'type', 'type': 'COMBO', 'widget': {'name': 'type'}},
            {'name': 'device', 'type': 'COMBO', 'widget': {'name': 'device'}},
        ],
        'outputs': [{'name': 'CLIP', 'type': 'CLIP', 'links': []}],
    },
    'AnimaBoosterLoader': {
        'inputs': [
            {'name': 'model_name', 'type': 'COMBO', 'widget': {'name': 'model_name'}},
            {'name': 'sage_attention', 'type': 'COMBO', 'widget': {'name': 'sage_attention'}},
            {'name': 'torch_compile', 'type': 'BOOLEAN', 'widget': {'name': 'torch_compile'}},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}],
    },
    'VAELoader': {
        'inputs': [
            {'name': 'vae_name', 'type': 'COMBO', 'widget': {'name': 'vae_name'}},
        ],
        'outputs': [{'name': 'VAE', 'type': 'VAE', 'links': []}],
    },
    'LoraLoaderModelOnly': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
            {'name': 'lora_name', 'type': 'COMBO', 'widget': {'name': 'lora_name'}},
            {'name': 'strength_model', 'type': 'FLOAT', 'widget': {'name': 'strength_model'}},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}],
    },
    'CLIPTextEncode': {
        'inputs': [
            {'name': 'clip', 'type': 'CLIP', 'link': None},
            {'name': 'text', 'type': 'STRING', 'widget': {'name': 'text', 'multiline': True}},
        ],
        'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': []}],
    },
    'AnimaTeaCache': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
            {'name': 'teacache_version', 'type': 'COMBO', 'widget': {'name': 'teacache_version'}},
            {'name': 'teacache_rel_l1', 'type': 'FLOAT', 'widget': {'name': 'teacache_rel_l1'}},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}],
    },
    'FLSamplerV4': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
            {'name': 'positive', 'type': 'CONDITIONING', 'link': None},
            {'name': 'negative', 'type': 'CONDITIONING', 'link': None},
            {'name': 'vae', 'type': 'VAE', 'link': None},
            {'name': 'latent_image', 'type': 'LATENT', 'link': None},
            {'name': 'seed', 'type': 'INT', 'widget': {'name': 'seed'}},
            {'name': 'steps', 'type': 'INT', 'widget': {'name': 'steps'}},
            {'name': 'cfg', 'type': 'FLOAT', 'widget': {'name': 'cfg'}},
            {'name': 'sampler_name', 'type': 'COMBO', 'widget': {'name': 'sampler_name'}},
            {'name': 'scheduler', 'type': 'COMBO', 'widget': {'name': 'scheduler'}},
            {'name': 'denoise', 'type': 'FLOAT', 'widget': {'name': 'denoise'}},
        ],
        'outputs': [{'name': 'LATENT', 'type': 'LATENT', 'links': []}],
    },
    'VAEDecode': {
        'inputs': [
            {'name': 'samples', 'type': 'LATENT', 'link': None},
            {'name': 'vae', 'type': 'VAE', 'link': None},
        ],
        'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
    },
    'SaveImage': {
        'inputs': [
            {'name': 'images', 'type': 'IMAGE', 'link': None},
            {'name': 'filename_prefix', 'type': 'STRING', 'widget': {'name': 'filename_prefix'}},
        ],
        'outputs': [],
    },
    'RtxVideoSuperResolution': {
        'inputs': [
            {'name': 'images', 'type': 'IMAGE', 'link': None},
            {'name': 'rtx_vsr_quality', 'type': 'COMBO', 'widget': {'name': 'rtx_vsr_quality'}},
        ],
        'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
    },
    'AnimaArtistPack': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
            {'name': 'clip', 'type': 'CLIP', 'link': None},
            {'name': 'artist_chain', 'type': 'STRING', 'widget': {'name': 'artist_chain', 'multiline': True}},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}, {'name': 'CLIP', 'type': 'CLIP', 'links': []}],
    },
    'AnimaArtistCrossAttn': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
            {'name': 'artist_chain', 'type': 'STRING', 'widget': {'name': 'artist_chain', 'multiline': True}},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}],
    },
    'AnimaLayerReplayPatcher': {
        'inputs': [
            {'name': 'model', 'type': 'MODEL', 'link': None},
        ],
        'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': []}],
    },
    'AnimaImageScaleToTotalPixels': {
        'inputs': [
            {'name': 'images', 'type': 'IMAGE', 'link': None},
            {'name': 'megapixels', 'type': 'FLOAT', 'widget': {'name': 'megapixels'}},
            {'name': 'upscale_method', 'type': 'COMBO', 'widget': {'name': 'upscale_method'}},
        ],
        'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
    },
    'PreviewImage': {
        'inputs': [
            {'name': 'images', 'type': 'IMAGE', 'link': None},
        ],
        'outputs': [],
    },
}

def convert(api_wf):
    nodes_list = []
    links_list = []
    next_link_id = 1
    
    # 先收集所有链接关系: (src_id, src_slot) -> [(dst_id, dst_input_name)]
    connections = {}
    input_to_src = {}  # (dst_id, input_name) -> (src_id, src_slot)
    
    for nid, node in api_wf.items():
        for iname, val in node.get('inputs', {}).items():
            if isinstance(val, list) and len(val) == 2:
                src_id, src_slot = str(val[0]), int(val[1])
                key = (src_id, src_slot)
                if key not in connections:
                    connections[key] = []
                connections[key].append((nid, iname))
                input_to_src[(nid, iname)] = (src_id, src_slot)
    
    # 构建节点
    used_link_ids = set()
    for nid in sorted(api_wf.keys(), key=int):
        node = api_wf[nid]
        ct = node.get('class_type', 'Unknown')
        inputs_config = node.get('inputs', {})
        defs = NODE_DEFS.get(ct, {'inputs': [], 'outputs': []})
        
        # 构建 inputs
        fw_inputs = []
        widgets_values = []
        for idef in defs['inputs']:
            iname = idef['name']
            ival = inputs_config.get(iname, None)
            
            inp = dict(idef)
            inp.pop('widget', None)
            
            if isinstance(ival, list) and len(ival) == 2:
                # 是连接
                src_id, src_slot = str(ival[0]), int(ival[1])
                link_id = next_link_id
                next_link_id += 1
                inp['link'] = link_id
                
                # 找对应的输出 slot index
                src_defs = NODE_DEFS.get(api_wf.get(src_id, {}).get('class_type', ''), {'outputs': []})
                out_slot = 0
                for si, odef in enumerate(src_defs['outputs']):
                    if odef['name'] == idef.get('related_output', '') or si == src_slot:
                        out_slot = si
                        break
                
                link = [link_id, int(src_id), out_slot, int(nid), len([x for x in fw_inputs if x.get('link') is not None]), idef.get('type', '*')]
                links_list.append(link)
                used_link_ids.add(link_id)
            else:
                inp['link'] = None
                if 'widget' in idef:
                    widgets_values.append(ival if ival is not None else '')
            
            fw_inputs.append(inp)
        
        # 构建 outputs
        fw_outputs = []
        for oi, odef in enumerate(defs['outputs']):
            out = dict(odef)
            key = (nid, oi)
            conns = connections.get(key, [])
            out['links'] = [l[0] for l in links_list if l[3] == int(nid) and l[2] == oi]  # we'll fix these
            out['slot_index'] = oi
            fw_outputs.append(out)
        
        # 处理没有输入连接的 widget 值
        if not widgets_values:
            for idef in defs['inputs']:
                if 'widget' in idef:
                    iname = idef['name']
                    ival = inputs_config.get(iname, '')
                    if not (isinstance(ival, list) and len(ival) == 2):
                        widgets_values.append(ival if ival is not None else '')
        
        node_obj = {
            'id': int(nid),
            'type': ct,
            'pos': [0, 0],  # will be filled later
            'size': [300, 100],
            'flags': {},
            'order': 0,
            'mode': 0,
            'inputs': fw_inputs,
            'outputs': fw_outputs,
            'properties': {'Node name for S&R': ct},
            'widgets_values': widgets_values if widgets_values else None,
        }
        nodes_list.append(node_obj)
    
    # 修复 outputs links
    for link in links_list:
        link_id, src_id, src_slot, dst_id, dst_slot, ltype = link
        for node in nodes_list:
            if node['id'] == src_id:
                for oi, out in enumerate(node['outputs']):
                    if oi == src_slot:
                        if out['links'] is None or len(out['links']) == 0:
                            out['links'] = [link_id]
                        else:
                            out['links'].append(link_id)
    
    # 自动布局
    # 根据拓扑排序分配位置
    col_map = {
        'CLIPLoader': 0, 'AnimaBoosterLoader': 0, 'VAELoader': 0,
        'LoraLoaderModelOnly': 1, 'CLIPTextEncode': 1,
        'AnimaTeaCache': 2,
        'FLSamplerV4': 3,
        'VAEDecode': 4,
        'RtxVideoSuperResolution': 5,
        'SaveImage': 6, 'PreviewImage': 6,
        'AnimaArtistPack': 1, 'AnimaArtistCrossAttn': 2,
        'AnimaLayerReplayPatcher': 2,
        'AnimaImageScaleToTotalPixels': 4,
    }
    
    col_counts = {}
    for node in nodes_list:
        col = col_map.get(node['type'], 0)
        col_counts[col] = col_counts.get(col, 0) + 1
    
    col_idx = {c: 0 for c in col_counts}
    for node in nodes_list:
        ct = node['type']
        col = col_map.get(ct, 0)
        row = col_idx[col]
        col_idx[col] = row + 1
        
        node['pos'] = [100 + col * 360, 100 + row * 200]
        node['size'] = [300, 150]
    
    # 填充 order (按拓扑)
    for i, node in enumerate(nodes_list):
        node['order'] = i
    
    # 构建最终输出
    max_id = max(int(k) for k in api_wf.keys()) if api_wf else 0
    max_link = max(l[0] for l in links_list) if links_list else 0
    
    result = {
        'id': None,
        'revision': 0,
        'last_node_id': max_id,
        'last_link_id': max_link + 1 if links_list else 1,
        'nodes': nodes_list,
        'links': links_list,
        'groups': [],
        'config': {},
        'extra': {'ds': {'scale': 1.0, 'offset': [0, 0]}},
        'version': 0.4,
    }
    
    return result

if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    
    with open(src, 'r', encoding='utf-8') as f:
        api_wf = json.load(f)
    
    web_wf = convert(api_wf)
    
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(web_wf, f, indent=2, ensure_ascii=False)
    
    print(f'Converted: {os.path.basename(src)} -> {os.path.basename(dst)} ({len(web_wf["nodes"])} nodes, {len(web_wf["links"])} links)')
