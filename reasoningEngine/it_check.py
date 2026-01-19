import os, sys
import json
import argparse
# import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
from reason_engine import DrivingLogicEngine

it_check_example = {
    "it_check": "satisfied or unsatisfied based on the CORRECT intention",
    "explanation": "The ego car should stop, because the front car is too close to the ego car."
}

def generate(query, img, model_name):
    try:
        model = genai.GenerativeModel(model_name) 
        response = model.generate_content([query, img])
        clean_text = response.text.strip('`').lstrip('json\n')
        return json.loads(clean_text)

    except Exception as e:
        print(f"An error occurred: {e}")

def main():

    # load_dotenv()
    # os.getenv("GOOGLE_API_KEY")
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_it_path', type=str, default='result.json')
    parser.add_argument('--intention_relation', type=str, default='synonym_action.json')
    parser.add_argument('--image_dir', default='../dataset/LingoQA/videos')
    parser.add_argument('--scene', default='lingoqa_gtasd.json')
    parser.add_argument('--rules', type=str, default='uk_rules.json')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model_name', default="gemini-2.5-flash")
    parser.add_argument('--intention', type=bool, default=False)

    args = parser.parse_args()

    scene_path = args.scene
    save_it_path = args.save_it_path
    intention_relation = args.intention_relation
    # symbolic_result = args.symbolic_result
    image_dir = args.image_dir
    verbose = args.verbose
    model_name = args.model_name
    rule_path = args.rules
    intention = args.intention


    with open(scene_path, 'r') as f:
        scenes = json.load(f)
    with open(intention_relation, 'r') as f:
        it_relation = json.load(f)
    with open(rule_path, 'r') as f:
        rules = json.load(f)
    
    key_intention_actions = [v for k, v in it_relation.items()]
    key_intention = [k for k, v in it_relation.items()]
    syno_actions = key_intention_actions + key_intention

    engine = DrivingLogicEngine(rules, verbose)
    result = {}
    for seg_id, scene in scenes.items():
        result[seg_id] = {}
        actions_to_take, intend_action = engine.reasoning(seg_id, scene)
        action_list = [d['action'] for d in actions_to_take]
        it_list = list(intend_action)
        # if len(it_list) > 1:
        #     print(f'Process break due to multiple intentions!')
        #     break
        # find the key intention from intention_relation
        # key_intention = [k for k, v in it_relation.items() if any(x in v for x in action_list)]
        for it in it_list:
            # if it not in action_list and it not in syno_actions:
            if intention: 
                # check the if intention satisfied
                video_name = seg_id + '.jpg'
                video_path = os.path.join(image_dir, video_name)
                try:
                    img = PIL.Image.open(video_path)
                except FileNotFoundError:
                    print(f"Error: The image file was not found at '{video_path}'.")
                    exit() 
                # check with LLM
                query = f"Please check if the original intention {it} of the ego car is satisfied based on the visual information from five continuous frames of a driving video. Please give the answer in json format and the result should include the answer ('satisfied' or 'unsatisfied') and the explanation. For example: {it_check_example}"
                
                it_result = generate(query, img, model_name) # json
                # if it_result["it_check"] == "satisfied":
                    # actions_to_take.append(it)
                result[seg_id]["actions_to_take"] = actions_to_take
                result[seg_id]["intention_check"] = it_result
                result[seg_id]["intention_check"]["intention"] = [i for i in it_list]
                
            else:
                result[seg_id]["actions_to_take"] = actions_to_take
                result[seg_id]["intention"] = [i for i in it_list]

            with open(save_it_path, "w") as f_log:
                json.dump(result, f_log, indent=4)
    




if __name__ == '__main__':
    main()