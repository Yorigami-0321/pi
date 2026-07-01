import json, os

def add_layout(workflow_path, output_path=None):
    with open(workflow_path, 'r', encoding='utf-8') as f:
        wf = json.load(f)
    
    # 节点标题映射
    node_titles = {
        'CLIPLoader': 'CLIP Loader',
        'AnimaBoosterLoader': 'Anima Booster',
        'VAELoader': 'VAE Loader',
        'LoraLoaderModelOnly': 'LoRA Loader',
        'CLIPTextEncode': 'CLIP Text Encode',
        'AnimaTeaCache': 'TeaCache',
        'FLSamplerV4': 'FL Sampler',
        'VAEDecode': 'VAE Decode',
        'SaveImage': 'Save Image',
        'RtxVideoSuperResolution': 'RTX VSR',
        'AnimaArtistPack': 'Artist Pack',
        'AnimaArtistCrossAttn': 'Artist CrossAttn',
        'AnimaLayerReplayPatcher': 'Layer Replay',
        'AnimaImageScaleToTotalPixels': 'Scale',
        'PreviewImage': 'Preview',
    }
    
    # 分析连接: 找出每个节点的入边
    output_to_inputs = {}
    in_degree = {nid: 0 for nid in wf if '_meta' not in wf[nid]}
    
    for nid, node in wf.items():
        if '_meta' in node:
            continue
        for input_name, val in node.get('inputs', {}).items():
            if isinstance(val, list) and len(val) == 2:
                src_id, src_output = val
                src_key = (str(src_id), src_output)
                if src_key not in output_to_inputs:
                    output_to_inputs[src_key] = []
                output_to_inputs[src_key].append((nid, input_name))
                in_degree[nid] = in_degree.get(nid, 0) + 1
    
    # 拓扑排序
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        nid = queue.pop(0)
        topo_order.append(nid)
        src_key = (nid, 0)
        for target_nid, _ in output_to_inputs.get(src_key, []):
            in_degree[target_nid] -= 1
            if in_degree[target_nid] == 0:
                queue.append(target_nid)
    
    if not topo_order:
        topo_order = [nid for nid in wf if '_meta' not in wf[nid]]
    
    # 按类型分列
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
    
    col_width = 360
    row_height = 180
    col_counts = {}
    
    # 先统计每列节点数
    for nid in topo_order:
        node = wf[nid]
        if '_meta' in node:
            continue
        ct = node.get('class_type', '')
        col = col_map.get(ct, 0)
        col_counts[col] = col_counts.get(col, 0) + 1
    
    # 分配位置
    col_idx = {c: 0 for c in col_counts}
    for nid in topo_order:
        node = wf[nid]
        if '_meta' in node:
            continue
        ct = node.get('class_type', '')
        col = col_map.get(ct, 0)
        row = col_idx[col]
        col_idx[col] = row + 1
        
        x = 100 + col * col_width
        y = 100 + row * row_height
        title = node_titles.get(ct, ct)
        
        node['_meta'] = {
            'title': title,
            'x': x,
            'y': y,
            'width': 300,
        }
    
    output = output_path or workflow_path
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
    
    print(f'Done: {os.path.basename(workflow_path)} -> {os.path.basename(output)}')

if __name__ == '__main__':
    import sys
    add_layout(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
