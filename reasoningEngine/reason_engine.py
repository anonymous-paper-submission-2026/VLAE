from itertools import product
import os
import argparse
import json
import re
import logging
import sys
from datetime import datetime

class DrivingLogicEngine:

    def __init__(self, rules, verbose):
        """
        Initializes the engine with a list of taxonomy and rules.

        Args:
            taxonomy (dict[str, list[str]]): A list of dictionaries 
            rules: Read from json file.
        """
        self.taxonomy = {
        "road_user": ["road_user", "car", "van", "bus", "truck", "motorcyclist", "cyclist", "pedestrian", "scooter"], 
        "vehicle": ["vehicle", "car","van"], 
        "large_vehicle": ["large_vehicle", "bus", "truck"],
        "vulnerable_road_user": ["vulnerable_road_user", "cyclist", "motorcyclist", "pedestrian", "scooter"]
} 
        self.rules = rules 
        self.organise()
        self.verbose = verbose

        # self.model_name = model_name
        self.logger, self.log_filepath = self.setup_logging()
     
        # if self.verbose:
        self.print_axiom_conditions()
        self.print_rules()

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}.log"
        
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_filepath = os.path.join(log_dir, log_filename)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
            
        return logger, log_filepath

    def infer_actions(self, facts):
        ans = []
        fact_cond = []
        for fact in facts: 
            if fact in self.axiom_condition_id:
                fact_cond.append(self.axiom_condition_id[fact])
                if self.verbose:
                    self.logger.info(f"({fact})\t as axiom condition id {self.axiom_condition_id[fact]}")
            elif self.verbose:
                self.logger.info(f"({fact})\t\t not in axiom condition id list")
        self.fact_cond = fact_cond
        fact_cond = sorted(fact_cond)

        if self.verbose:
            self.logger.info(f"\nFact conditions ids: {fact_cond}\n")

        action_dict_list = [self.rule_dict]
        
        # print('***start to search action***')
        for fact_id in fact_cond:
            # print(fact_id)
            for d in action_dict_list:
                if fact_id in d:
                    action_dict_list.append(d[fact_id])
        
        for d in action_dict_list:
            if 'action' in d:
                ans.append(d)

        # if 'ego, at, junction' in facts and 
        if ('traffic_light, was, red' in facts or 'traffic_light, was, amber' in facts) and 'traffic_light, is, green' in facts:
            ans.append({'rule_id': '58', 'action': 'start'})
        
        # Hierachy
        # the following rules have higher hierachy than rule 55: maintain speed
        over_55 = {1,2,3,4,5,6,7,8,12,16,24,25,27,28,32,33,44,45,46,47,48,49,50,51,52,53,54, 58, 63}
        
        fired_rule = set([int(i['rule_id']) for i in ans])

        if len(over_55 & fired_rule) == 0:
            ans.append({'rule_id': '70', 'action': 'maintain_speed'})

        setA = {1, 8, 19, 20} # stop by traffic light
        setB = {2, 14, 15, 16, 17, 37, 38} # proceed with turning light

        if any(int(item.get('rule_id')) in setA for item in ans):
            ans = [item for item in ans if int(item.get('rule_id')) not in setB]

        
                
        return ans
    

    def organise(self):
        axiom_conditions = set()
        conditions_action = {}

        self.rule_dict = {}
        self.action_conditions = {}

        # ---------- taxonomy expansion ----------
        for rule in self.rules:
            rule_id = int(rule['id'])
            action = rule['action']
            condition_list = rule['conditions']

            candidate_conditions = []
            valid = True

            for axiom_cond in condition_list:
                mapped = self.taxonomy_reasoning(axiom_cond)
                if not mapped:
                    valid = False
                    break
                axiom_conditions.update(mapped)
                candidate_conditions.append(mapped)

            if not valid:
                continue

            if candidate_conditions:
                mapped_conditions = [list(p) for p in product(*candidate_conditions)]
            else:
                mapped_conditions = [[]]

            conditions_action[(rule_id, action)] = mapped_conditions
            self.action_conditions.setdefault(action, []).append({
                'rule_id': rule_id,
                'conditions': mapped_conditions
            })

        # ---------- axiom id maps ----------
        self.axiom_condition_id = {
            item: idx for idx, item in enumerate(axiom_conditions)
        }
        self.id_axiom_conditions = {
            idx: item for idx, item in enumerate(axiom_conditions)
        }

        # ---------- build nested dict-only rule tree ----------
        self.rule_dict = {}

        for (rule_id, action), condition_list in conditions_action.items():
            for condition in condition_list:
                if not condition:
                    continue

                cond_id_list = sorted(
                    self.axiom_condition_id[c] for c in condition
                )

                curr = self.rule_dict
                for cid in cond_id_list[:-1]:
                    if cid not in curr or not isinstance(curr[cid], dict):
                        curr[cid] = {}
                    curr = curr[cid]

                leaf = cond_id_list[-1]

                curr[leaf] = {
                    'rule_id': rule_id,
                    'action': action
                }


    
    def build_rule_dict(self, curr_dict, cond_id):
        cond_id_dict = curr_dict.setdefault(cond_id, {})
        return cond_id_dict


    def taxonomy_reasoning(self, condition):
        # replace the general class ('vulnerable road users') by the child-classes ('pedestrain')
        # then product the classes to generate all possible conditions
        condition = [item.strip() for item in condition.split(",")]
        candidates = [
            self.taxonomy.get(item, [item])  # keep original if there is no related terms
            for item in condition
        ]

        tax_reasoned_conditions = [", ".join(list(p)) for p in product(*candidates)]
        return tax_reasoned_conditions # list of possible conditions (in list)  
    

    def print_rules(self):
        self.logger.info("\nFollowing are the rules")
        for rule in self.rule_dict:
            self.logger.info(f"{rule}, {self.rule_dict[rule]}")

    def print_axiom_conditions(self):
        self.logger.info("\nFollowing are the axiom_condition with ids")
        for c in self.id_axiom_conditions:
            self.logger.info(f"\t{c}: {self.id_axiom_conditions[c]}")

    
    def reasoning(self, scene_id, scene_discription):
        situation = scene_discription["situation"]
        control_device = scene_discription['control_device']
        road_user = scene_discription['road_user']
        intention = scene_discription['intention']

        facts = [tri.strip('()') for tri in situation]
        for statement in control_device:
            device, _, previous_state, current_state = [i.strip() for i in statement.strip('()').split(',')]

            facts.append(f"{device}, is, exist")
            facts.append(f"{device}, status, exist")
            facts.append(f"{device}, was, {previous_state}") 
            facts.append(f'{device}, is, {current_state}') # "traffic_light is green"
            facts.append(f'{device}, status, {current_state}') # "traffic_light is green"

        ego_being_overtaken = 0
        for statement in road_user:
            try:
                user, position, previous_state, current_state = [i.strip() for i in statement.strip('()').split(',')]
            except AttributeError:
                print(statement)
            # facts.append(f"road_user, {position}, ego")
            # facts.append(f"road_user, is, {current_state}")
            # facts.append(f"{user}, is, exist")
            facts.append(f"road_user, {position}, ego")
            facts.append(f"road_user, is, {current_state}")
            facts.append(f"road_user, status, {current_state}")
            facts.append(f"{user}, {position}, {current_state}")
            facts.append(f"{user}, {position}, ego")
            facts.append(f"{user}, is, exist")
            facts.append(f"{user}, status, exist")
            facts.append(f"{user}, status, {current_state}")
            if position == "in_front_of" or position == "same_lane_relevant":
                facts.append(f"ego, approaching, {user}")
                facts.append(f"{user}, same_lane_front_of, ego")
                facts.append(f"road_user, same_lane_front_relevant, ego")
            if current_state == "overtake_ego":
                ego_being_overtaken = 1
        if not ego_being_overtaken:
            facts.append(f"road_users, are, not_begin_overtake_ego")

        intended_action = set()
        for statement in intention:
            _, intent = [i.strip() for i in statement.strip('()').split(',')]
            if intent.startswith("turn"):
                facts.append("ego, intend, turn")
            # else:
            facts.append(f"ego, intend, {intent}")
            intended_action.add(intent)

        if self.verbose:
            self.logger.info(f"\n\nfacts for scene {scene_id}:")
            for f in facts:
                self.logger.info(f'\t{f}')
            self.logger.info('\n')
        
        reasoning_result = self.infer_actions(facts)  

        reasoning_result = [{'rule_id': i['rule_id'], 'action': i['action']} for i in reasoning_result if 'action' in i and 'rule_id' in i]

        # checked = self.check_intentions(intended_action)

        # logger.info(f'\nintention check: {checked}')
        # if self.verbose:
        self.logger.info(f"\n\nReasoning results for scene {scene_id}:")
        self.logger.info(f"\n\t\tActions: {reasoning_result}")
        self.logger.info(f"actions: {set([a['action'] for a in reasoning_result])}")

        return reasoning_result, intended_action