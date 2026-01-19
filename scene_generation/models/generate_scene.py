import os
import json
import re
import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
from datetime import datetime
import argparse
import pandas as pd

class Scene:
    def __init__(self, model_name, vocab_path):
        self.example_change_description = {
            "situation": [
                "(ego, in, residential_area)",
                "(ego, passing, road_work)",
                "(ego, has_crossed, white_line)"
            ],
            "control_device": [
                "(traffic_light, relevant, red, green)",
            ],
            "road_user": [
                "(silver car, same_lane_front_of, stopped, moving_forward)",
                "(red van, left_lane, moving_forward, moving_forward)",
                "(pedestrian, left_sidewalk, walk_away, walk_away)",
                "(red bus, left_lane, invisible, overtak_ego)",
                "(oncoming_traffic, different_lane_front_of, exist, invisible)",
                "(black car, very_close_to, crossing, crossing)",
                "(white car, far_way, moving_forward, moving_forward)"
            ],
            "intention": [
                "(ego, moving_forward)"
            ]
            }

        self.f_situation = "(ego, ego_situation)"
        self.f_control_device = "(control_device, relevant, previous status, current status)"
        self.f_road_user = "(road_user, road_user_position, previous_status, current_status)"
        self.f_intention = "(ego, ego_intention)"

        self.example_scene_graph = {
        "frame_id": 4,
        "ego_situation": [
            "(is, stopped)",
            "(approaching, crossing)",
            "(in, buildup_area)"
        ],
        "road_users": [
            {
            "type": "pedestrian",
            "position": "right_sidewalk",
            "status": "walk_away"
            }
        ],
        "road_features": [
            {
            "type": "crossing",
            "status": "clear"
            }
        ]
        }

        with open(vocab_path, 'r') as f:
            self.vocabulary = json.load(f)

        self.model_name = model_name
        load_dotenv()
        os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def extract_words(self, vocab):
        words = set()
        if isinstance(vocab, dict):
            for v in vocab.values():
                words |= self.extract_words(v)
        elif isinstance(vocab, list):
            for item in vocab:
                if isinstance(item, str):
                    # Extract individual words from phrases like "(passing, crossing)"
                    words |= set(re.findall(r'\b\w+\b', item))
                else:
                    words |= self.extract_words(item)
        elif isinstance(vocab, str):
            words.add(vocab)
        return words

    def generate(self, query, img=None):
        try:
        # Initialize the generative model
            model = genai.GenerativeModel(self.model_name) 
            if img:
                response = model.generate_content([query, img])
            else:
                response = model.generate_content([query])

            # --- Output and Validation ---
            response_words = response.text.lower()
            clean_text = response_words.strip('`').lstrip('json\n')
            res_words = json.loads(clean_text)
            res_set = self.extract_words(res_words)
            vocab_set = self.extract_words(self.vocabulary)

            for word in ['ego', 'relevant']:
                res_set.discard(word)
            # res_set.discard('ego')
            print("Validation Check:")
            if res_set.issubset(vocab_set):
                print("Success! The response uses only words from the defined vocabulary.")
                return res_words
            else:
                invalid_words = res_set - vocab_set
                print(f"Warning! The response contains words not in the vocabulary: {', '.join(invalid_words)}")
                return res_words
            
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_scene(self, image_dir, video_ids, ref_video_ids, save_path):
        for video in video_ids:
            if video in ref_video_ids:
                try:
                    with open(save_path, 'r') as f_log:
                        scene_data = json.load(f_log)
                except (FileNotFoundError,json.JSONDecodeError):
                # If the file is empty, start with an empty dictionary
                    scene_data = {}
                while video not in list(scene_data.keys()) or scene_data[video] is None:
                    
                    video_name = video + '.jpg'
                    video_path = os.path.join(image_dir, video_name)
                    try:
                        img = PIL.Image.open(video_path)
                    except FileNotFoundError:
                        print(f"Error: The image file was not found at '{video_path}'.")
                        exit() 
                        
                    
                    query = f"""
                    Please describe the scene of a driving video (five frames) from four perspectives: **Situation, Control_device, Road_user, and Intention**.  

                    Guidelines:  
                    1. **Situation**:  
                    - Describe the situation of the EGO car in the **last frame only**.  
                    - Use the format: {self.f_situation}.  

                    2. **Control_device**:  
                    - Conclude the status or status changes of relevant control devices **for the ego car’s lane only**.  
                    - Ignore control devices not relevant to the ego’s lane.  
                    - Use the format: {self.f_control_device}.  
                    - Include both `previous_status` (summarising frames 1–4) and `current_status` (frame 5).  

                    3. **Road_user**:  
                    - Describe the moving status of all road users across frames.  
                    - `previous_status` summarises frames 1–4, `current_status` is from frame 5.  
                    - `road_user_position` gives the relative position of each road user to the ego car.  
                    - Use the format: {self.f_road_user}.  

                    4. **Intention**:  
                    - Describe the driving intention of the ego car.  
                    - Use the format: {self.f_intention}.  
                    - All turning directions (left/right) must be from the **ego car’s perspective** and decided carefully based on surrounding reference objects.  

                    5. **Vocabulary constraint**:  
                    - Use **only** words and combinations of words from the vocabulary: {self.vocabulary}.  
                    - Do not introduce new objects, relations, or movements not present in the video.  

                    6. **Output format**:  
                    - Provide the answer strictly in **JSON format**.  
                    - Follow the example structure exactly: {self.example_change_description}.  

                    """
 
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Generating ASD for video: {video}")
                    result = self.generate(query, img) # json
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Video: {video} aggregated scene description generated")

                    scene_item = {video: result}
                    
                    # save to json file
                    scene_data.update(scene_item)

                    with open(save_path, "w") as f_log:
                        json.dump(scene_data, f_log, indent=4)
                    
                    print(f'{video} done')

    def get_tkg(self, image_dir, video_ids, ref_video_ids, save_path):
        for video in video_ids:
            if video in ref_video_ids:
                try:
                    tkgpath = save_path.replace('.json', '-tkg.json')
                    with open(tkgpath, 'r') as tkg_log:
                        tkg_data = json.load(tkg_log)
                        if video in tkg_data.keys():
                            ctkg_data = tkg_data[video]
                except (FileNotFoundError,json.JSONDecodeError):   
                # If the file is empty, start with an empty dictionary
                    tkg_data = {}   

                while video not in list(tkg_data.keys()) or tkg_data[video] is None:
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Generating TKG for video: {video}")
                    video_name = video + '.jpg'
                    video_path = os.path.join(image_dir, video_name)
                    try:
                        img = PIL.Image.open(video_path)
                    except FileNotFoundError:
                        print(f"Error: The image file was not found at '{video_path}'.")
                        exit() 
                        

                    query = f"""
                    You are a professional driving assistant. Given a sequence of five consecutive frames extracted from a driving video.
                    Your task is to generate **five scene graphs** in **JSON format**, one per frame, capturing the dynamic spatial and temporal relations relevant for driving decisions.

                    Please follow these detailed instructions:

                    1. **Scene focus:** Identify all visible and relevant objects (e.g., ego car, other vehicles, pedestrians, cyclists, traffic lights, road signs). Focus on objects and relations that influence driving decisions.
                    2. **Temporal coherence:** The five frames may appear visually similar, but reflect gradual motion or state changes. 
                    Ensure your scene graphs show small, realistic temporal differences between frames (e.g., a car approaching, pedestrian starting to cross, light turning green).
                    3. **Directional reference:** All directions (e.g., left, right, front, behind) must be described **from the ego car’s perspective**. Determine turning directions carefully using spatial references (e.g., lanes, intersections, other vehicles).
                    4. **Vocabulary constraint:** You must **only use words or word combinations** from the following controlled vocabulary:
                    {self.vocabulary}
                    Do **not** invent new objects, attributes, or relations not included in the vocabulary.
                    5. **Output format:** Provide the output as valid JSON with one list entry per frame.
                    Each frame should be a scene graph following the structure shown below:
                    {self.example_scene_graph}

                    Additional rules:
                    - Be consistent in object naming across frames (e.g., "vehicle_1" refers to the same car if it persists).
                    - Avoid hallucinations: only include objects, relations, or movements that can be visually inferred from the video.
                    - Ensure that relations (e.g., `same_lane_front_relevant`, `approaching`, `status`) are physically plausible and consistent over time.

                    Now, generate the five scene graphs in JSON format.
                    """

                    result = self.generate(query, img) # json
                    tkg_item = {video: result}
                    ctkg_data = tkg_item[video]
                    # save to json file
                    tkg_data.update(tkg_item)
                    # self.tkg_data = tkg_data

                    with open(tkgpath, "w") as tkg_log:
                        json.dump(tkg_data, tkg_log, indent=4)

                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Video: {video} TKG generated")

                try:
                    with open(save_path, 'r') as f_log:
                        scene_data = json.load(f_log)
                except (FileNotFoundError,json.JSONDecodeError):
                # If the file is empty, start with an empty dictionary
                    scene_data = {}

                while video not in list(scene_data.keys()) or scene_data[video] is None:
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} generating ASD for video: {video}")
                    video_name = video + '.jpg'
                    video_path = os.path.join(image_dir, video_name)


                    query = f"""
                    You are a professional driving assistant. Based on the five scene graphs here {ctkg_data}, please summarise the scene of a driving video (five frames) from four perspectives: **Situation, Control_device, Road_user, and Intention**.  

                    Guidelines:  
                    1. **Situation**:  
                    - Describe the situation of the EGO car in the **last frame only**.  
                    - Use the format: {self.f_situation}.  

                    2. **Control_device**:  
                    - Conclude the status or status changes of relevant control devices **for the ego car’s lane only**.  
                    - Ignore control devices not relevant to the ego’s lane.  
                    - Use the format: {self.f_control_device}.  
                    - Include both `previous_status` (summarising frames 1–4) and `current_status` (frame 5).  

                    3. **Road_user**:  
                    - Describe the moving status of all road users across frames.  
                    - `previous_status` summarises frames 1–4, `current_status` is from frame 5.  
                    - `road_user_position` gives the relative position of each road user to the ego car.  
                    - Use the format: {self.f_road_user}.  

                    4. **Intention**:  
                    - Describe the driving intention of the ego car.  
                    - Use the format: {self.f_intention}.  
                    - All turning directions (left/right) must be from the **ego car’s perspective** and decided carefully based on surrounding reference objects.  

                    5. **Vocabulary constraint**:  
                    - Use **only** words and combinations of words from the vocabulary: {self.vocabulary}.  
                    - Do not introduce new objects, relations, or movements not present in the video.  

                    6. **Output format**:  
                    - Provide the answer strictly in **JSON format**.  
                    - Follow the example structure exactly: {self.example_change_description}.  

                    """

                    result = self.generate(query) # json
                    scene_item = {video: result}
                    # save to json file
                    scene_data.update(scene_item)
                    with open(save_path, "w") as f_log:
                        json.dump(scene_data, f_log, indent=4)
                    
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Video: {video} aggregated scene description generated")