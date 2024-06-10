import time
from flask import Flask, request, jsonify
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

model_path = './models/sam_vit_h_4b8939.pth'
model_type = 'vit_h'
sam = sam_model_registry[model_type](checkpoint=model_path)
sam.to(device='cuda')

app = Flask(__name__)


def load_image(image_url):
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)


def generate_masks(image_url):
    image = load_image(image_url)
    begin_time = time.time()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )
    masks = mask_generator.generate(image)
    print(f"Time taken: {time.time() - begin_time}")
    return process_masks(masks)


def process_masks(masks):
    with ThreadPoolExecutor(max_workers=4) as executor:
        masks = list(executor.map(encode_positions, [masks[i] for i in range(len(masks))]))
    return convert_to_json(convert_np_to_py(masks))


def convert_np_to_py(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_np_to_py(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_to_py(element) for element in data]
    return data


def convert_to_json(data):
    formatted_mask_data = [
        [(int(start), int(length)) for start, length in mask]
        for mask in data
    ]
    return formatted_mask_data


def encode_positions(mask):
    mask_flat = mask.flatten()
    start_indices = np.where((mask_flat[:-1] == 0) & (mask_flat[1:] == 1))[0] + 1
    end_indices = np.where((mask_flat[:-1] == 1) & (mask_flat[1:] == 0))[0] + 1
    if mask_flat[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if mask_flat[-1] == 1:
        end_indices = np.append(end_indices, len(mask_flat))
    lengths = end_indices - start_indices
    positions = list(zip(start_indices, lengths))
    return positions


@app.route('/api/process-url', methods=['POST'])
def process_url():
    url = request.json.get('url')
    mask_data = generate_masks(url)
    result = {'status': 'success', "encodeMask": mask_data, "img": url}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8004)