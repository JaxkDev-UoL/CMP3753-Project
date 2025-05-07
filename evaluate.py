import json
import os
import random
import re
import time
from collections import defaultdict
from typing import List, Dict

from inference import generate_response, get_model

#random.seed(42)  # For reproducibility

TEST_CASES = [
    {
        "input": "<<wave>>",
        "valid_response_patterns": [
            {"required": ["<<wave>>"], "optional": ["<<speak>>"]},
            {"required": ["<<speak>>"], "optional": []}
        ]
    },
    {
        "input": "<<speak>> Show me your best item!",
        "valid_response_patterns": [
            {"required": ["<<offer>>", "<item>", "<price>"], "optional": ["<<speak>>", "<currency>"]},
            {"required": ["<<speak>>"], "optional": ["<<grumble>>"]},
            {"required": ["<<grumble>>"], "optional": ["<<offer>>", "<item>", "<price>"]}
        ]
    },
    {
        "input": "<<speak>> I'll take 3!",
        "valid_response_patterns": [
            {"required": ["<<give>>", "<item>"], "optional": ["<<speak>>"]}
        ]
    }
]

class TokenMetrics:
    def __init__(self):
        self.total_interactions = 0
        self.token_counts = defaultdict(int)
        self.pattern_matches = defaultdict(int)
        self.token_stats = {
            'correct': defaultdict(int),
            'missing': defaultdict(int),
            'extra': defaultdict(int)
        }
        
    def update(self, detected: List[str], expected_patterns: List[Dict]):
        self.total_interactions += 1
        best_match = self._find_best_match(detected, expected_patterns)
        
        for token in detected:
            self.token_counts[token] += 1
            if token in best_match['matched']:
                self.token_stats['correct'][token] += 1
            else:
                self.token_stats['extra'][token] += 1
                
        for token in best_match['expected']:
            if token not in detected:
                self.token_stats['missing'][token] += 1
                
        self.pattern_matches[best_match['pattern_id']] += 1

    def _find_best_match(self, detected, patterns):
        best = {'score': -1, 'pattern_id': None, 'matched': [], 'expected': []}
        for idx, pattern in enumerate(patterns):
            required = set(pattern['required'])
            optional = set(pattern['optional'])
            expected = required.union(optional)
            
            detected_set = set(detected)
            matched_required = required.intersection(detected_set)
            score = len(matched_required)/len(required) if required else 1
            
            if score > best['score']:
                best = {
                    'score': score,
                    'pattern_id': idx,
                    'matched': list(detected_set.intersection(expected)),
                    'expected': list(expected)
                }
        return best

def evaluate_responses(llm_responses: List[str], test_cases: List[Dict]) -> Dict:
    metrics = TokenMetrics()
    token_pattern = re.compile(r'(<<.*?>>|<.*?>)')
    
    for response, test_case in zip(llm_responses, test_cases):
        detected_tokens = token_pattern.findall(response)
        metrics.update(detected_tokens, test_case["valid_response_patterns"])
    
    return metrics

class MockLLM:
    def generate(self, input_text):
        if "<<wave>>" in input_text:
            return random.choice([
                "<<wave>>",
                "<<wave>>\n<<speak>> Greetings!",
                "<<speak>> Hello there!"
            ])
        elif "best item" in input_text:
            return random.choice([
                "<<offer>> <item> Dragon Slayer <price> 999",
                "<<speak>> Nothing special today",
                "<<grumble>>\n<<offer>> <item> Rusty Sword <price> 50"
            ])
        elif "I'll take 3" in input_text:
            return random.choice([
                "<<give>> <item> 3 Potions",
                "<<speak>> Out of stock!",
                "<<give>> <item> 3 Scrolls\n<<speak>> Enjoy!"
            ])
        return "<<speak>> I don't understand"

class RealLLM:
    def generate(self, input_text):
        return generate_response([{
            "role": "user",
            "content": input_text
        }])

def run_evaluation(num_runs=100):
    llm = MockLLM()
    llm = RealLLM()  # Uncomment this line to use the real LLM
    selected_test_cases = [random.choice(TEST_CASES) for _ in range(num_runs)]

    model = get_model()

    from tqdm import tqdm
    llm_responses = []
    for i in tqdm(range(num_runs), desc="Evaluating"):
        llm_responses.append(llm.generate(selected_test_cases[i]["input"]))
    
    if not os.path.exists("evaluation_results"):
        os.makedirs("evaluation_results")
    
    timestamp = time.strftime("%d%m%y-%H%M%S")
    with open(f"evaluation_results/results_{timestamp}.jsonl", "w") as f:
        f.write(json.dumps({"timestamp": timestamp, "model": model}) + "\n")
        for i in range(num_runs):
            f.write(json.dumps({'input': selected_test_cases[i]['input'], 'output': llm_responses[i]}) + "\n")
    
    metrics = evaluate_responses(llm_responses, selected_test_cases)
    
    total_tokens = sum(metrics.token_counts.values())
    token_precision = sum(metrics.token_stats['correct'].values()) / total_tokens
    token_recall = sum(metrics.token_stats['correct'].values()) / (
        sum(metrics.token_stats['correct'].values()) + 
        sum(metrics.token_stats['missing'].values())
    )
    
    return {
        "token_level": {
            "precision": token_precision,
            "recall": token_recall if (sum(metrics.token_stats['correct'].values()) + sum(metrics.token_stats['missing'].values()) > 0) else 1.0,
            "f1": 2 * (token_precision * token_recall) / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0
        },
        "interaction_level": {
            "full_match_rate": sum(metrics.pattern_matches.values()) / metrics.total_interactions,
            "partial_match_rate": sum(metrics.pattern_matches.values()) / metrics.total_interactions
        },
        "error_analysis": {
            "most_missing": max(metrics.token_stats['missing'], key=lambda k: metrics.token_stats['missing'][k], default="None"),
            "most_extra": max(metrics.token_stats['extra'], key=lambda k: metrics.token_stats['extra'][k], default="None"),
            "error_distribution": {
                "missing": dict(metrics.token_stats['missing']),
                "extra": dict(metrics.token_stats['extra'])
            }
        },
        "pattern_effectiveness": {
            "best_pattern": max(metrics.pattern_matches, key=metrics.pattern_matches.get, default=-1),
            "pattern_distribution": dict(metrics.pattern_matches)
        }
    }

if __name__ == "__main__":
    results = run_evaluation(num_runs=1000)
    
    print("Numerical Evaluation Results (100 Random Samples):\n")
    print(f"Token Precision: {results['token_level']['precision']:.2%}")
    print(f"Token Recall: {results['token_level']['recall']:.2%}")
    print(f"Token F1: {results['token_level']['f1']:.2%}\n")
    
    print(f"Full Pattern Match Rate: {results['interaction_level']['full_match_rate']:.2%}")
    print(f"Partial Match Rate: {results['interaction_level']['partial_match_rate']:.2%}\n")
    
    print("Error Analysis:")
    print(f"Most Missing Token: {results['error_analysis']['most_missing']}")
    print(f"Most Extra Token: {results['error_analysis']['most_extra']}")
    print("\nError Distribution:")
    print("Missing:", results['error_analysis']['error_distribution']['missing'])
    print("Extra:", results['error_analysis']['error_distribution']['extra'])
    
    print("\nPattern Effectiveness:")
    print(f"Best Pattern: {results['pattern_effectiveness']['best_pattern']}")
    print("Distribution:", results['pattern_effectiveness']['pattern_distribution'])