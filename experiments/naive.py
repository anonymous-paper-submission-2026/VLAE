import os
import json
import re
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
import argparse
import pandas as pd
from datetime import datetime

def main():
    def generate_answer(query, img, model_name):
        try:
            model = genai.GenerativeModel(model_name) 
            response = model.generate_content([query, img])
            clean_text = response.text.strip('`').lstrip('json\n')
            return json.loads(clean_text)
        except Exception as e:
            print_log(f"An error occurred: {e}")    
    def print_log(message):
        with open(log_path, 'a') as log_file:
            log_file.write(message + '\n')
        print(message)

    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='../images/path')
    parser.add_argument('--qae_file', type=str, default='data/lingoqa/val.parquet')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model_name', default="gemini-2.5-flash")
    args = parser.parse_args()

    image_dir = args.image_dir
    save_path = args.save_path
    qae_file = args.qae_file
    model_name = args.model_name
    
    imagebatch = os.path.basename(image_dir).split('_')[0]
    modelsuffix = model_name.split('-')[-1]
    save_path = f'1-naive/lingoqa_naive_{imagebatch}_{modelsuffix}_1001.json'    
    log_path = save_path.replace('.json', '.log')

    # get all images with ground truth qae
    df = pd.read_parquet(qae_file)
    unique_video = df['segment_id'].unique().tolist()

    videos = os.listdir(image_dir)
    video_ids = [os.path.splitext(f)[0] for f in videos if f.endswith('.jpg')]
    for video in video_ids:
        if video in unique_video:
            try:
                with open(save_path, 'r') as f_log:
                    result_data = json.load(f_log)
            except (FileNotFoundError,json.JSONDecodeError):
                result_data = {}
            if video not in list(result_data.keys()):
                video_name = video + '.jpg'
                video_path = os.path.join(image_dir, video_name)
                try:
                    img = PIL.Image.open(video_path)
                except FileNotFoundError:
                    print_log(f"Error: The image file was not found at '{video_path}'.")
                    exit() 
                query = f"please answer the question for the driving video: what is the best action to take for the ego car? The result should be in json format with three keys: 'action', 'explanation' and 'summary'. The 'action' should be the necessary actions to take as short phrases. The 'explanation' should be detailed explanation and be concise. The 'summary' should be a one sentence explanation of the final actions."

                print_log(f"{datetime.now():%Y-%m-%d %H:%M:%S} Generating answer for video: {video}")
                result = generate_answer(query, img, model_name) # json
                print_log(f"{datetime.now():%Y-%m-%d %H:%M:%S} Generating answer for video: {video}")
                result_item = {video: result}

                # save to json file
                result_data.update(result_item)
                
                with open(save_path, "w") as f_log:
                    json.dump(result_data, f_log, indent=4)
                print_log(f'{video} done')

if __name__ == '__main__':
    main()