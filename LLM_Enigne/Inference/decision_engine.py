import json
import time
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class ThreatAssessment:
    threat_level:         str
    threat_score:         float
    confidence:           float
    reasoning:            str
    recommended_actions:  List[str]
    timestamp:            float

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# PROMPT TEMPLATES
# ============================================================

class PromptTemplates:

    SYSTEM_PROMPT = """You are an AI security analyst for an autonomous surveillance system.

You must respond ONLY with valid JSON in this exact format:

{
    "threat_level": "low|medium|high|critical",
    "threat_score": 0.0,
    "confidence": 0.0,
    "reasoning": "brief explanation",
    "recommended_actions": ["action1", "action2"]
}

Do not include explanations outside JSON.
"""

    @staticmethod
    def create_assessment_prompt(context: Dict) -> str:
        return f"""
Analyze the following surveillance context:

Detections: {context.get('detections', [])}
Primary Object: {context.get('primary_object', 'unknown')}
Number of Objects: {context.get('num_detections', 0)}

Zone: {context.get('zone', 'unknown')}
Restricted Area: {context.get('restricted_area', False)}
Time of Day: {context.get('time_of_day', 'unknown')}

Motion Type: {context.get('motion_type', 'unknown')}
Speed: {context.get('speed', 'unknown')}
Loitering: {context.get('loitering', False)}
Direction Changes: {context.get('direction_changes', 0)}
Time in Area: {context.get('time_in_area', 0)} seconds

Audio Events: {context.get('audio_events', [])}
Vibration: {context.get('vibration', False)}
LiDAR Objects: {context.get('lidar_objects', 0)}

Previous Alerts: {context.get('previous_alerts', 0)}
Unusual Pattern: {context.get('unusual_pattern', False)}
"""


# ============================================================
# DECISION ENGINE
# ============================================================

class LLMDecisionEngine:

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path:    str = r"C:\Users\smrut\Desktop\projects\Project 1 -ASTAS Autonomous Surveillance & Threat Assessment System\qlora-qwen2.5-1.5b-astas",
        device:          str = "auto",
    ):
        self.model     = None
        self.tokenizer = None

        try:
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,        # 'dtype' replaces deprecated 'torch_dtype'
                device_map=device,
                trust_remote_code=True,
            )

            print("Loading QLoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )

            self.model.eval()
            print("LLM loaded successfully.")

        except Exception as e:
            print("Failed to load LLM. Falling back to rule-based system.")
            print("Error:", e)

        self.prompt_templates = PromptTemplates()

        self.threat_thresholds = {
            'critical': 0.85,
            'high':     0.65,
            'medium':   0.40,
            'low':      0.0,
        }

        # Internal history for aggregate reporting
        self._assessment_history: List[ThreatAssessment] = []

    # ============================================================
    # PUBLIC INTERFACE
    # ============================================================

    def assess_threat(self, context: Dict) -> ThreatAssessment:
        if self.model is not None:
            try:
                assessment = self._llm_assessment(context)
            except Exception as e:
                print("LLM failed, using rule-based fallback:", e)
                assessment = self._rule_based_assessment(context)
        else:
            assessment = self._rule_based_assessment(context)

        self._assessment_history.append(assessment)
        return assessment

    def generate_report(
        self,
        assessment: Optional[ThreatAssessment] = None,
        context:    Optional[Dict]             = None,
    ) -> str:
        """
        Generate a human-readable report string.

        Called by main.py as:  engine.generate_report(assessment, ctx)

        Parameters
        ----------
        assessment : ThreatAssessment (or list of them), optional
            The assessment(s) to include.  If None, uses the full history.
        context : dict, optional
            The raw sensor context dict — printed as supplementary info.

        Returns
        -------
        str
            Formatted multi-line report.
        """
        # Accept a single assessment OR a list
        if assessment is None:
            source = self._assessment_history
        elif isinstance(assessment, list):
            source = assessment
        else:
            source = [assessment]

        if not source:
            return "  [Report] No assessments available."

        lines = ["", "  ┌─── Threat Report ───────────────────────────────────"]

        level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        highest = max(source, key=lambda a: level_order.get(a.threat_level.lower(), 0))
        avg_score = sum(a.threat_score for a in source) / len(source)

        lines.append(f"  │  Assessments    : {len(source)}")
        lines.append(f"  │  Highest threat : {highest.threat_level.upper()}")
        lines.append(f"  │  Avg score      : {avg_score:.3f}")
        lines.append(f"  │  Confidence     : {highest.confidence:.2f}")
        lines.append(f"  │  Reasoning      : {highest.reasoning}")

        actions = highest.recommended_actions
        lines.append(f"  │  Actions        : {', '.join(actions)}")

        if context:
            lines.append(f"  │  Zone           : {context.get('zone', 'N/A')}")
            lines.append(f"  │  Restricted     : {context.get('restricted_area', False)}")
            lines.append(f"  │  Time of day    : {context.get('time_of_day', 'N/A')}")

        lines.append("  └─────────────────────────────────────────────────────")
        return "\n".join(lines)

    def clear_history(self):
        self._assessment_history.clear()

    # ============================================================
    # LLM ASSESSMENT
    # ============================================================

    def _llm_assessment(self, context: Dict) -> ThreatAssessment:
        messages = [
            {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT},
            {"role": "user",   "content": self.prompt_templates.create_assessment_prompt(context)},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                 **inputs,
                 max_new_tokens=256,
                 do_sample=True,
                 top_p=0.9,
            )

        decoded  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_data = self._extract_json(decoded)

        return ThreatAssessment(
            threat_level=json_data["threat_level"],
            threat_score=float(json_data["threat_score"]),
            confidence=float(json_data["confidence"]),
            reasoning=json_data["reasoning"],
            recommended_actions=json_data["recommended_actions"],
            timestamp=time.time(),
        )

    # ============================================================
    # ROBUST JSON EXTRACTION
    # ============================================================

    def _extract_json(self, text: str) -> Dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")

        parsed = json.loads(match.group())

        for field in ["threat_level", "threat_score", "confidence",
                      "reasoning", "recommended_actions"]:
            if field not in parsed:
                raise ValueError(f"Missing field: {field}")

        return parsed

    # ============================================================
    # RULE-BASED FALLBACK
    # ============================================================

    def _rule_based_assessment(self, context: Dict) -> ThreatAssessment:
        threat_score    = 0.0
        reasoning_parts = []
        actions         = []

        if context.get('restricted_area', False):
            threat_score += 0.3
            reasoning_parts.append("Restricted area breach")

        if context.get('time_of_day') == 'night':
            threat_score += 0.2
            reasoning_parts.append("Night activity")

        if context.get('loitering', False):
            threat_score += 0.25
            reasoning_parts.append("Loitering detected")

        if context.get('unusual_pattern', False):
            threat_score += 0.1
            reasoning_parts.append("Unusual movement pattern")

        if 'gunshot' in context.get('audio_events', []):
            threat_score    = 1.0
            reasoning_parts = ["Gunshot detected"]
            actions         = ["IMMEDIATE ALERT", "Contact authorities"]

        threat_score = min(threat_score, 1.0)

        if threat_score >= self.threat_thresholds['critical']:
            level = 'critical'
        elif threat_score >= self.threat_thresholds['high']:
            level = 'high'
        elif threat_score >= self.threat_thresholds['medium']:
            level = 'medium'
        else:
            level = 'low'

        if not actions:
            actions = ["Monitor situation"]

        return ThreatAssessment(
            threat_level=level,
            threat_score=threat_score,
            confidence=0.7,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Normal activity",
            recommended_actions=actions,
            timestamp=time.time(),
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    engine = LLMDecisionEngine()

    test_context = {
        'detections': ['person'], 'num_detections': 3,
        'primary_object': 'person', 'zone': 'red',
        'restricted_area': True, 'time_of_day': 'day',
        'motion_type': 'rapid_movement', 'speed': 'fast',
        'loitering': True, 'direction_changes': 6,
        'time_in_area': 90, 'audio_events': [],
        'vibration': False, 'lidar_objects': 3,
        'previous_alerts': 1, 'unusual_pattern': True,
    }

    assessment = engine.assess_threat(test_context)
    print("\nThreat Assessment:")
    print(json.dumps(assessment.to_dict(), indent=2))
    print(engine.generate_report(assessment, test_context))