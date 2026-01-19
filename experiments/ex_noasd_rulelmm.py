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
    parser.add_argument('--image_dir', default='../images/path')
    parser.add_argument('--qae_file', type=str, default='data/lingoqa/val.parquet')
    parser.add_argument('--rules',default='uk_rules.json')
    parser.add_argument('--model_name', default="gemini-2.5-flash")

    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    qae_file = args.qae_file
    model_name = args.model_name
    rules = args.rules    
    save_path = args.save_path

    imagebatch = os.path.basename(image_dir).split('_')[0]
    modelsuffix = model_name.split('-')[-1]
    save_path = f'5-noasd_rulelmm/lingoqa_{imagebatch}_{modelsuffix}_1001_newquery.json'    
    log_path = save_path.replace('.json', '.log')

    df = pd.read_parquet(qae_file)
    unique_video = df['segment_id'].unique().tolist()

    with open(rules, 'r') as f:
        rules = json.load(f)
    
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
                    print_log( f"Error: The image file was not found at '{video_path}'.")
                    exit() 
                
                # query = f"please answer the question for the driving video (consists of five frames): what is the best action to take for the ego car? The answer should be based on the visual information from the video and the UK traffic rules {rules}. You should first retrieve relevant rules based on the visual information, and then reason over these rules to find the best action and explain the reasoning process using triggered rules. ONLY trigger the rule if all conditions in the rule are satisfied. For example, 'conditions': ['ego, approaching, vulnerable_road_user', 'vulnerable_road_user, same_lane_front_of, ego'], the rule should be triggered if two conditions in the list are satisfied. If no rule is triggered, the reasoning path should follow rule 55. If the previous status of ego is stop and the current status is move_off, or the previous traffic light is red and the current traffic light is green, the reasoning path should follow rule 56. Then, verify if the ego car’s intention in this video is covered by the retrieved rules; if not, check whether this intention is still allowed under the rules for this action. Finally, rank the reasoning path based on the priority of the rules, decide the order of the best actions, and remove the contradictory and unnecessary actions. The result should be in json format with four keys: 'action', 'reasoning_path', 'explanation' and 'summary'. The value of 'reasoning_path' contains the conditions (if several conditions exist) and the corresponding action of a rule, and put several reasoning paths (if exist) in a list, such as 'reasoning_path': [('ego, approaching, vulnerable_road_user', 'vulnerable_road_user, same_lane_front_of, ego', 'reduce_speed'), ('ego, on, motorway', 'must_not_reverse'), ...]. The 'explanation' value should be based on the reasoning_path and be concise. The value of 'summary' should be a one sentence explanation of the final actions, for example: 'The best action is to ..., because ...' Output JSON in this format: {jsonformat}. Here is an example: {example}"

                query = f"""You are a driving assistant. Answer the question: "What is the best action for the ego car?" 
                Use the video (5 frames) and the UK traffic rules {rules}.
                Follow this process step by step:
                1. Retrieve all rules that may apply. 
                2. Check whether all conditions of each rule are satisfied by video information. 
                - A rule is triggered only if ALL its conditions hold. 
                - If no rule triggers, apply Rule (Default Behaviour). 
                - If ego changed from stop to move_off, apply Rule (Common Sense).
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
                print_log( f'{video} done')

if __name__ == '__main__':
    main()
 
