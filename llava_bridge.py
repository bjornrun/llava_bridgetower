import argparse
import logging
import time
from io import BytesIO
import PIL.Image
import requests
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BridgeTowerProcessor,
    BridgeTowerForImageAndTextRetrieval,
    pipeline
)
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

app = Flask(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global variables to store model and configuration

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

generator = None
generate_kwargs = None

def initialize_model(args):
    global generator, generate_kwargs

    adapt_transformers_to_gaudi()

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    generator = pipeline(
        "image-to-text",
        model=args.model_name_or_path,
        torch_dtype=model_dtype,
        device="hpu",
    )

    generate_kwargs = {
        "lazy_mode": True,
        "hpu_graphs": args.use_hpu_graphs,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        generator.model = wrap_in_hpu_graph(generator.model)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    image_url = data.get('image_url')
    prompt = data.get('prompt')

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        image = PIL.Image.open(requests.get(image_url, stream=True, timeout=3000).raw)
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

    start = time.perf_counter()
    result = generator([image], prompt=prompt, batch_size=1, generate_kwargs=generate_kwargs)
    end = time.perf_counter()
    duration = end - start

    return jsonify({
        "result": result,
        "time_taken": duration * 1000,  # Convert to milliseconds
    })
    
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text and image URL from the request
    data = request.json
    texts = data.get('texts', [])
    image_url = data.get('image_url')

    if not isinstance(texts, list):
        return jsonify({"error": "texts must be an array of strings"}), 400


    if not texts or not image_url:
        return jsonify({"error": "Both texts and image_url are required"}), 400

    try:
    # Process the image
        image = Image.open(requests.get(image_url, stream=True).raw)

        results = []
        for text in texts:
            encoding = processor(image, text, return_tensors="pt")
            outputs = model(**encoding)
            results.append(outputs.logits[0,1].item())


        return jsonify({
            "results": results,
            "image_url": image_url
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to pre-trained model")
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Whether to use HPU graphs or not.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument("--ignore_eos", action="store_true", help="Whether to ignore eos, set False to disable it.")
    args = parser.parse_args()

    initialize_model(args)
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
    
