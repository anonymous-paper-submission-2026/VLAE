import os, sys
import json
import re
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
import argparse
import pandas as pd
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from models.generate_scene import Scene


def main():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='../images/path')
    parser.add_argument('--qae_file', type=str, default='data/lingoqa/val.parquet')
    parser.add_argument('--vocabulary', default='vocabulary.json')
    parser.add_argument('--model_name', default="gemini-2.5-pro")
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    save_path = args.save_path
    qae_file = args.qae_file
    model_name = args.model_name
    vocab_path = args.vocabulary

    imagebatch = os.path.basename(image_dir).split('_')[0]
    modelsuffix = model_name.split('-')[-1]
    save_path = f'scene_result/lingoqa_asdlmm_{imagebatch}_{modelsuffix}_1004.json'
    # get all images with ground truth qae
    df = pd.read_parquet(qae_file) 
    unique_video = df['segment_id'].unique().tolist() # get all unique image/segment ids

    videos = os.listdir(image_dir) 
    video_ids = [os.path.splitext(f)[0] for f in videos if f.endswith('.jpg')] # get all existent image ids

    generate_scene = Scene(model_name, vocab_path) # initialise scene generation model
    generate_scene.get_scene(image_dir, video_ids, unique_video, save_path) # generate scene for all images and save to save_path

if __name__ == '__main__':
    main()