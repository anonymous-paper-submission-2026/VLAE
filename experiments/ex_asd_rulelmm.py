import os, logging
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
 
    jsonformat = """{
            "reasoning_path": [
                {"UKRuleid": "Rule X", "id": N, "conditions": [...], "action": "..."}
            ],
            "action": [list of best actions as short phrases],
            "explanation": "Concise explanation with rule IDs.",
            "summary": "One sentence summary of best action and why."
            }"""
    example = """{
            "reasoning_path": [
                {"UKRuleid": "Rule 2", "id": 58, "conditions": ["ego, approaching, vulnerable_road_user", "vulnerable_road_user, same_lane_front_of, ego"], "action": "reduce_speed"}
            ],
            "action": ["reduce_speed"],
            "explanation": "According to UK traffic Rule 2 (id 58), since a vulnerable road user is in front of ego, the ego must reduce speed.",
            "summary": "The best action is to reduce speed, because a vulnerable road user is ahead."
            }
            """
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument('--image_dir', default='../images/path')
    parser.add_argument('--scene_path', default='scene.json')
    parser.add_argument('--qae_file', type=str, default='data/lingoqa/val.parquet')
    parser.add_argument('--rules',default='uk_rules.json')
    parser.add_argument('--model_name', default="gemini-2.5-flash")
    # outputs
    parser.add_argument('--save_path', type=str, default=None)
    
    args = parser.parse_args()
    image_dir = args.image_dir
    scene_path = args.scene_path
    qae_file = args.qae_file
    rules = args.rules
    model_name = args.model_name
    save_path = args.save_path

    # auto set paths
    # imagebatch = os.path.basename(image_dir).split('_')[0]
    # scene_path = f'scene_result/gt_asd_{imagebatch}.json'
    scenefilename = 'tkgasd'#(os.path.basename(scene_path)).split('.')[0]
    modelsuffix = model_name.split('-')[-1]
    save_path = f'6-asd_rulelmm/lingoqa_{scenefilename}_rulelmm_{modelsuffix}_1004_newquery.json'    
    log_path = save_path.replace('.json', '.log')



    # get all images with ground truth qae
    df = pd.read_parquet(qae_file)
    unique_video = df['segment_id'].unique().tolist()

    with open(rules, 'r') as f:
        rules = json.load(f)
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
            while video not in list(result_data.keys()) or result_data[video] is None:
                
                video_name = video + '.jpg'
                video_path = os.path.join(image_dir, video_name)
                try:
                    img = PIL.Image.open(video_path)
                except FileNotFoundError:
                    print_log(f"Error: The image file was not found at '{video_path}'.")
                    exit() 
                scene_description = scene[video]
                query = f"""You are a driving assistant. Answer the question: "What is the best action for the ego car?" 
                Use the video (5 frames), the scene description {scene_description}, and the UK traffic rules {rules}.
                Follow this process step by step:
                1. Retrieve all rules that may apply. 
                2. Check whether all conditions of each rule are satisfied by the scene description. 
                - A rule is triggered only if ALL its conditions hold. 
                - If no rule triggers, apply Rule 64 (Default Behaviour). 
                - If ego changed from stop to move_off, apply Rule 65 (Common Sense).
                3. Resolve conflicts: if two rules contradict, keep the one with higher priority  according to the priority rules.
                4. Create the final reasoning path, actions, explanation, and summary.
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
                print_log(f'{video} done')


if __name__ == '__main__':
    main()