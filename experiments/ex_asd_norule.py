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
    def print_log(message):
        with open(log_path, 'a') as log_file:
            log_file.write(message + '\n')
        print(message)
    def generate_answer(query, img, model_name):
        try:
            model = genai.GenerativeModel(model_name) 
            response = model.generate_content([query, img])
            clean_text = response.text.strip('`').lstrip('json\n')
            return json.loads(clean_text)

        except Exception as e:
            print_log( f"An error occurred: {e}")

    jsonformat = {
            "action": "list of best actions as short phrases",
            "explanation": "Concise explanation based on video and scene description.",
            "summary": "One sentence summary of best action and why."
            }
    example = {
        "action": [
                "moving_forward",
                "drive_carefully_and_slowly"
            ],
        "explanation": "The ego should drive carefully and slowly because it is approaching a crossing. The traffic light is green therefore it can move forward.",
        "summary": "The best action is to ..., because ..."
    }    
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='../images/path')
    parser.add_argument('--qae_file', type=str, default='data/lingoqa/val.parquet')
    parser.add_argument('--scene_path', default='scene.json')
    parser.add_argument('--model_name', default="gemini-2.5-flash")
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    save_path = args.save_path
    qae_file = args.qae_file
    model_name = args.model_name
    scene_path = args.scene_path
    
    scenefilename = (os.path.basename(scene_path)).split('.')[0]
    modelsuffix = model_name.split('-')[-1]
    save_path = f'4-asd_norule/lingoqa_{scenefilename}_rulelmm_{modelsuffix}_1001_newquery.json'    
    log_path = save_path.replace('.json', '.log')

    # get all images with ground truth qae
    df = pd.read_parquet(qae_file)
    unique_video = df['segment_id'].unique().tolist()

    with open(scene_path, 'r') as f:
        scene = json.load(f)

    videos = os.listdir(image_dir)
    video_ids = [os.path.splitext(f)[0] for f in videos if f.endswith('.jpg')]
    for video in video_ids:
        if video in unique_video:
            try:
                with open(save_path, 'r') as f_log:
                    result_data = json.load(f_log)
            except (FileNotFoundError,json.JSONDecodeError):
                result_data = {}
            while video not in list(result_data.keys()):
                video_name = video + '.jpg'
                video_path = os.path.join(image_dir, video_name)
                try:
                    img = PIL.Image.open(video_path)
                except FileNotFoundError:
                    print_log( f"Error: The image file was not found at '{video_path}'.")
                    exit() 
                scene_description = scene[video]

                query = f"""You are a driving assistant. Answer the question: "What is the best action for the ego car?" 
                Use the video (5 frames) and the scene description {scene_description},  
                Output JSON in this format: {jsonformat}. Here is an example: {example}
                """

                print_log(f"{datetime.now():%Y-%m-%d %H:%M:%S} Generating answer for video: {video}")
                result = generate_answer(query, img, model_name) # json
                print_log(f"{datetime.now():%Y-%m-%d %H:%M:%S} Answer generated for video: {video}")
                result_item = {video: result}

                # save to json file
                result_data.update(result_item)

                with open(save_path, "w") as f_log:
                    json.dump(result_data, f_log, indent=4)
                print_log( f'{video} is done.')

if __name__ == '__main__':
    main()