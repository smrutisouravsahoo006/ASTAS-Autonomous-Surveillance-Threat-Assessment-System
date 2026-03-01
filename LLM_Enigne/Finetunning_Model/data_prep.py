import os
import sys
import json
import random
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ASTAS.DataPrep")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
DPO_DIR    = DATA_DIR / "dpo"
STATS_DIR  = DATA_DIR / "stats"

for d in [RAW_DIR, PROC_DIR, DPO_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Domain vocabulary  (mirrors SimulationContext fields)
# ─────────────────────────────────────────────────────────────────────────────

ZONES = ["restricted", "caution", "safe", "perimeter", "parking", "lobby", "rooftop", "server_room"]
OBJECT_TYPES = ["person", "vehicle", "drone", "group_of_people", "unknown_object", "animal"]
MOTION_TYPES = ["static", "slow", "normal", "rapid", "erratic", "retreating", "approaching"]
TIME_OF_DAY = ["day", "night", "dawn", "dusk"]
AUDIO_EVENTS = ["glass_break", "gunshot", "shouting", "alarm", "footsteps", "engine", "none"]
ENVIRONMENTS = ["border", "building_interior", "parking_lot", "warehouse", "airport", "campus", "datacenter"]
THREAT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Scenario templates
THREAT_SCENARIOS = {
    "LOW": [
        "Authorized personnel moving through a public corridor",
        "Single person walking at normal pace in safe zone",
        "Vehicle parked in designated area during business hours",
        "Security guard conducting a routine patrol sweep",
        "Delivery personnel near building entrance with valid credentials",
        "Employee accessing cafeteria during lunch break",
        "Visitor following an escort through common areas",
    ],
    "MEDIUM": [
        "Unidentified individual moving slowly near caution zone boundary",
        "Vehicle circling the parking lot multiple times without stopping",
        "Person photographing facility exterior from public area",
        "Unknown individual attempting to access staff-only stairwell",
        "Group gathering near perimeter fence at dusk",
        "Abandoned bag detected near caution zone entrance",
        "Person making repeated trips between two areas with no clear purpose",
    ],
    "HIGH": [
        "Individual breached perimeter fence and entered restricted zone",
        "Two unidentified persons loitering near server room for over 10 minutes",
        "Vehicle with obscured plates observed near critical infrastructure",
        "Person detected attempting to disable a security camera",
        "Rapid movement toward restricted zone after attempted access denial",
        "Unauthorized drone detected hovering above restricted airspace",
        "Individual observed at multiple access points in succession",
        "Person detected carrying concealed object near high-security area",
    ],
    "CRITICAL": [
        "Armed individual confirmed inside restricted zone, moving toward critical assets",
        "Multiple intruders breached perimeter simultaneously, coordinated movement detected",
        "Explosive device suspected — unattended bag vibration sensor triggered in server room",
        "Active threat: Person engaged security personnel and gained unauthorized access",
        "Multiple cameras disabled simultaneously, blind spot exploitation underway",
        "Drone swarm coordinating above sensitive facility after hours",
        "Intruder at critical infrastructure panel, manual override attempt detected",
        "Hostage situation in progress, panic button activated in restricted zone",
    ]
}

RECOMMENDED_ACTIONS = {
    "LOW":      ["Continue monitoring", "Log event for audit", "No immediate action required"],
    "MEDIUM":   ["Dispatch security guard to investigate", "Increase camera monitoring frequency",
                 "Issue verbal warning", "Request identification from individual"],
    "HIGH":     ["Dispatch rapid response team immediately", "Lock down affected zone",
                 "Notify shift supervisor and initiate alert protocol",
                 "Activate secondary surveillance on perimeter"],
    "CRITICAL": ["IMMEDIATE lockdown — all zones", "Engage law enforcement — call 911",
                 "Evacuate personnel from affected sectors",
                 "Activate emergency broadcast system",
                 "Deploy all available security assets to incident location"],
}

REASONING_TEMPLATES = {
    "LOW": [
        "No zone violations detected. Movement pattern is consistent with authorized access. "
        "Time of day and object type match expected activity. Threat score: {score:.2f}.",
        "Entity identified in safe zone exhibiting normal movement. No behavioral anomalies. "
        "No prior alerts in history. Low confidence of hostile intent.",
    ],
    "MEDIUM": [
        "Subject is near a caution zone boundary with {motion} movement. "
        "Time in area: {time:.0f}s — exceeds normal transit time. Requires investigation.",
        "Unidentified {obj} detected. Zone classification: {zone}. "
        "No credential match. Pattern suggests purposeful reconnaissance. Score: {score:.2f}.",
    ],
    "HIGH": [
        "Restricted area breach confirmed. {obj} entered {zone} zone without authorization. "
        "Loitering: {loitering}. Rapid movement detected: {rapid}. "
        "Historical alert count: {alerts}. Immediate intervention required.",
        "Multiple threat indicators active: zone violation + {motion} movement + sensor fusion "
        "confirms {camera} camera detections and {lidar} LiDAR objects. "
        "Threat score: {score:.2f} — escalating to HIGH.",
    ],
    "CRITICAL": [
        "CRITICAL: {num_det} entities in restricted zone. Coordinated movement pattern. "
        "Audio events: {audio}. All sensors confirm active intrusion. "
        "Score: {score:.2f}. Emergency protocols must activate immediately.",
        "Extreme threat: unauthorized access to {zone}. Armed/dangerous classification. "
        "Response delay increases risk exponentially. All available units required.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreatSample:
    """Single labeled threat assessment sample"""

    # Input context (mirrors SimulationContext)
    timestamp:          float
    time_of_day:        str
    environment:        str
    zone:               str
    restricted_area:    bool
    primary_object:     str
    num_detections:     int
    motion_type:        str
    speed:              float
    loitering:          bool
    rapid_movement:     bool
    direction_changes:  int
    time_in_area:       float
    camera_detections:  int
    lidar_objects:      int
    audio_events:       List[str]
    vibration:          bool
    previous_alerts:    int
    unusual_pattern:    bool

    # Ground-truth labels
    threat_level:       str
    threat_score:       float
    reasoning:          str
    recommended_action: str
    scenario_desc:      str

    # Metadata
    sample_id:          str = field(default="")
    source:             str = field(default="synthetic")
    created_at:         str = field(default="")

    def __post_init__(self):
        if not self.sample_id:
            raw = json.dumps(asdict(self), sort_keys=True).encode()
            self.sample_id = hashlib.md5(raw).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DPOPair:
    """Preference pair for DPO training"""
    prompt:   str
    chosen:   str   # Correct/better response
    rejected: str   # Incorrect/worse response
    sample_id: str  = field(default="")

    def __post_init__(self):
        if not self.sample_id:
            raw = (self.prompt + self.chosen).encode()
            self.sample_id = hashlib.md5(raw).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Sample Generator
# ─────────────────────────────────────────────────────────────────────────────

class RawSampleGenerator:
    """
    Synthesizes diverse, realistic threat assessment samples covering
    all ASTAS threat levels with controlled class balance.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        log.info("RawSampleGenerator initialized (seed=%d)", seed)

    # ── Score computation (mirrors ASTASBridge._fallback_assessment) ──────────

    def _compute_score(self, s: Dict) -> float:
        """Compute deterministic threat score from context fields"""
        score = 0.0
        if s["restricted_area"]:      score += 0.30
        if s["loitering"]:            score += 0.25
        if s["rapid_movement"]:       score += 0.20
        if s["num_detections"] > 5:   score += 0.15
        if s["unusual_pattern"]:      score += 0.15
        if s["previous_alerts"] > 2:  score += 0.10
        if s["vibration"]:            score += 0.10
        if s["time_in_area"] > 300:   score += 0.10
        if s["direction_changes"] > 4: score += 0.05
        audio_weight = len([a for a in s["audio_events"] if a != "none"]) * 0.05
        score += audio_weight
        # Zone modifier
        zone_mod = {"restricted": 0.15, "caution": 0.05, "safe": -0.05, "perimeter": 0.08}
        score += zone_mod.get(s["zone"], 0.0)
        return float(np.clip(score, 0.0, 1.0))

    def _score_to_level(self, score: float) -> str:
        if score >= 0.75: return "CRITICAL"
        if score >= 0.50: return "HIGH"
        if score >= 0.30: return "MEDIUM"
        return "LOW"

    # ── Reasoning builder ─────────────────────────────────────────────────────

    def _build_reasoning(self, s: Dict, level: str, score: float) -> str:
        templates = REASONING_TEMPLATES[level]
        tmpl = random.choice(templates)
        try:
            return tmpl.format(
                score=score,
                obj=s["primary_object"],
                zone=s["zone"],
                motion=s["motion_type"],
                loitering=s["loitering"],
                rapid=s["rapid_movement"],
                alerts=s["previous_alerts"],
                camera=s["camera_detections"],
                lidar=s["lidar_objects"],
                audio=", ".join(s["audio_events"]) or "none",
                time=s["time_in_area"],
                num_det=s["num_detections"],
            )
        except KeyError:
            return f"Threat score {score:.2f} classified as {level} based on sensor fusion analysis."

    # ── Context generators by threat level ───────────────────────────────────

    def _gen_low(self) -> Dict:
        return {
            "time_of_day":       random.choice(["day", "dawn"]),
            "environment":       random.choice(ENVIRONMENTS),
            "zone":              random.choice(["safe", "lobby", "parking"]),
            "restricted_area":   False,
            "primary_object":    random.choice(["person", "vehicle"]),
            "num_detections":    random.randint(1, 3),
            "motion_type":       random.choice(["slow", "normal"]),
            "speed":             round(random.uniform(0.5, 2.0), 2),
            "loitering":         False,
            "rapid_movement":    False,
            "direction_changes": random.randint(0, 1),
            "time_in_area":      round(random.uniform(5.0, 60.0), 1),
            "camera_detections": random.randint(0, 2),
            "lidar_objects":     random.randint(0, 2),
            "audio_events":      ["none"],
            "vibration":         False,
            "previous_alerts":   0,
            "unusual_pattern":   False,
        }

    def _gen_medium(self) -> Dict:
        return {
            "time_of_day":       random.choice(TIME_OF_DAY),
            "environment":       random.choice(ENVIRONMENTS),
            "zone":              random.choice(["caution", "perimeter", "parking"]),
            "restricted_area":   False,
            "primary_object":    random.choice(OBJECT_TYPES),
            "num_detections":    random.randint(1, 4),
            "motion_type":       random.choice(["slow", "normal", "erratic"]),
            "speed":             round(random.uniform(0.5, 3.5), 2),
            "loitering":         random.choice([True, False]),
            "rapid_movement":    False,
            "direction_changes": random.randint(2, 4),
            "time_in_area":      round(random.uniform(60.0, 300.0), 1),
            "camera_detections": random.randint(1, 3),
            "lidar_objects":     random.randint(0, 2),
            "audio_events":      random.choices(AUDIO_EVENTS, k=random.randint(0, 1)),
            "vibration":         False,
            "previous_alerts":   random.randint(0, 1),
            "unusual_pattern":   random.choice([True, False]),
        }

    def _gen_high(self) -> Dict:
        return {
            "time_of_day":       random.choice(["night", "dusk", "dawn"]),
            "environment":       random.choice(ENVIRONMENTS),
            "zone":              random.choice(["restricted", "perimeter", "rooftop"]),
            "restricted_area":   True,
            "primary_object":    random.choice(["person", "group_of_people", "drone"]),
            "num_detections":    random.randint(2, 6),
            "motion_type":       random.choice(["rapid", "erratic", "approaching"]),
            "speed":             round(random.uniform(3.0, 6.0), 2),
            "loitering":         random.choice([True, True, False]),
            "rapid_movement":    True,
            "direction_changes": random.randint(3, 7),
            "time_in_area":      round(random.uniform(120.0, 600.0), 1),
            "camera_detections": random.randint(2, 5),
            "lidar_objects":     random.randint(1, 4),
            "audio_events":      random.choices(["footsteps", "shouting", "glass_break", "none"], k=2),
            "vibration":         random.choice([True, False]),
            "previous_alerts":   random.randint(1, 4),
            "unusual_pattern":   True,
        }

    def _gen_critical(self) -> Dict:
        return {
            "time_of_day":       random.choice(["night", "dusk"]),
            "environment":       random.choice(["datacenter", "airport", "building_interior", "warehouse"]),
            "zone":              random.choice(["restricted", "server_room"]),
            "restricted_area":   True,
            "primary_object":    random.choice(["person", "group_of_people", "unknown_object"]),
            "num_detections":    random.randint(4, 12),
            "motion_type":       random.choice(["rapid", "erratic", "approaching"]),
            "speed":             round(random.uniform(5.0, 10.0), 2),
            "loitering":         True,
            "rapid_movement":    True,
            "direction_changes": random.randint(5, 10),
            "time_in_area":      round(random.uniform(300.0, 1200.0), 1),
            "camera_detections": random.randint(4, 8),
            "lidar_objects":     random.randint(3, 8),
            "audio_events":      random.choices(["gunshot", "glass_break", "alarm", "shouting"], k=2),
            "vibration":         True,
            "previous_alerts":   random.randint(3, 10),
            "unusual_pattern":   True,
        }

    # ── Main generation ───────────────────────────────────────────────────────

    def generate(self, n: int, balance: bool = True) -> List[ThreatSample]:
        """
        Generate n samples. If balance=True each threat level gets ~25%.
        """
        generators = {
            "LOW":      self._gen_low,
            "MEDIUM":   self._gen_medium,
            "HIGH":     self._gen_high,
            "CRITICAL": self._gen_critical,
        }

        samples: List[ThreatSample] = []

        if balance:
            per_level = n // 4
            remainder = n - per_level * 4
            counts = {lvl: per_level for lvl in THREAT_LEVELS}
            counts["CRITICAL"] += remainder
        else:
            # Natural distribution (more LOW than CRITICAL)
            weights = [0.35, 0.30, 0.25, 0.10]
            level_counts = np.random.multinomial(n, weights)
            counts = dict(zip(THREAT_LEVELS, level_counts.tolist()))

        for level, count in counts.items():
            gen_fn = generators[level]
            scenarios = THREAT_SCENARIOS[level]
            actions = RECOMMENDED_ACTIONS[level]

            for i in range(count):
                ctx = gen_fn()
                score = self._compute_score(ctx)

                # Clamp score to match level bucket
                ranges = {
                    "LOW":      (0.00, 0.29),
                    "MEDIUM":   (0.30, 0.49),
                    "HIGH":     (0.50, 0.74),
                    "CRITICAL": (0.75, 1.00),
                }
                lo, hi = ranges[level]
                score = float(np.clip(score, lo, hi))
                # Add small noise for realism
                score = float(np.clip(score + np.random.uniform(-0.02, 0.02), lo, hi))

                reasoning = self._build_reasoning(ctx, level, score)

                sample = ThreatSample(
                    timestamp=round(random.uniform(0, 86400), 1),
                    threat_level=level,
                    threat_score=round(score, 4),
                    reasoning=reasoning,
                    recommended_action=random.choice(actions),
                    scenario_desc=random.choice(scenarios),
                    **ctx
                )
                samples.append(sample)

        random.shuffle(samples)
        log.info("Generated %d samples: %s", len(samples), counts)
        return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt Formatter
# ─────────────────────────────────────────────────────────────────────────────

class PromptFormatter:
    """
    Converts ThreatSample → chat-formatted messages dict.
    Supports:
    - SFT (supervised fine-tuning): system + user + assistant
    - DPO (preference): prompt + chosen + rejected
    """

    SYSTEM_PROMPT = (
        "You are ASTAS — an Autonomous Surveillance Threat Assessment System. "
        "Your task is to analyze sensor data from a surveillance environment and "
        "produce a structured JSON threat assessment.\n\n"
        "Output format (strict JSON, no markdown):\n"
        '{\n'
        '  "threat_level": "LOW | MEDIUM | HIGH | CRITICAL",\n'
        '  "threat_score": <float 0.0-1.0>,\n'
        '  "reasoning": "<detailed explanation>",\n'
        '  "recommended_action": "<specific action>",\n'
        '  "confidence": "LOW | MEDIUM | HIGH"\n'
        '}'
    )

    def _context_to_user_message(self, s: ThreatSample) -> str:
        """Serialize ThreatSample context as a structured user prompt"""
        audio_str = ", ".join(s.audio_events) if s.audio_events else "none"
        lines = [
            "=== SENSOR FUSION REPORT ===",
            f"Timestamp      : {s.timestamp:.1f}s",
            f"Time of Day    : {s.time_of_day}",
            f"Environment    : {s.environment}",
            "",
            "--- Detection Data ---",
            f"Primary Object : {s.primary_object}",
            f"Total Detected : {s.num_detections}",
            f"Camera Hits    : {s.camera_detections}",
            f"LiDAR Objects  : {s.lidar_objects}",
            f"Audio Events   : {audio_str}",
            f"Vibration      : {'YES' if s.vibration else 'NO'}",
            "",
            "--- Location & Zone ---",
            f"Zone           : {s.zone.upper()}",
            f"Restricted Area: {'YES ⚠' if s.restricted_area else 'NO'}",
            "",
            "--- Behavior Analysis ---",
            f"Motion Type    : {s.motion_type}",
            f"Speed          : {s.speed:.1f} m/s",
            f"Loitering      : {'YES ⚠' if s.loitering else 'NO'}",
            f"Rapid Movement : {'YES ⚠' if s.rapid_movement else 'NO'}",
            f"Dir. Changes   : {s.direction_changes}",
            f"Time in Area   : {s.time_in_area:.0f}s",
            f"Unusual Pattern: {'YES ⚠' if s.unusual_pattern else 'NO'}",
            "",
            "--- History ---",
            f"Previous Alerts: {s.previous_alerts}",
            "",
            "--- Scenario Context ---",
            f"{s.scenario_desc}",
            "",
            "Provide your threat assessment as strict JSON.",
        ]
        return "\n".join(lines)

    def _build_assistant_response(self, s: ThreatSample) -> str:
        """Build the ideal JSON assistant response"""
        confidence_map = {"LOW": "MEDIUM", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "CRITICAL": "HIGH"}
        response = {
            "threat_level": s.threat_level,
            "threat_score": round(s.threat_score, 4),
            "reasoning": s.reasoning,
            "recommended_action": s.recommended_action,
            "confidence": confidence_map[s.threat_level],
        }
        return json.dumps(response, indent=2)

    def to_sft(self, sample: ThreatSample) -> Dict:
        """Convert sample to SFT chat format"""
        return {
            "sample_id": sample.sample_id,
            "messages": [
                {"role": "system",    "content": self.SYSTEM_PROMPT},
                {"role": "user",      "content": self._context_to_user_message(sample)},
                {"role": "assistant", "content": self._build_assistant_response(sample)},
            ],
            "metadata": {
                "threat_level": sample.threat_level,
                "threat_score": sample.threat_score,
                "source": sample.source,
                "created_at": sample.created_at,
            }
        }

    def to_alpaca(self, sample: ThreatSample) -> Dict:
        """Convert to Alpaca instruction format (alternative to chat)"""
        return {
            "instruction": self.SYSTEM_PROMPT,
            "input": self._context_to_user_message(sample),
            "output": self._build_assistant_response(sample),
        }

    def to_dpo(self, sample: ThreatSample, wrong_level: str) -> Optional[DPOPair]:
        """
        Build a DPO preference pair.
        chosen  = correct assessment
        rejected = plausible-but-wrong assessment (wrong threat level)
        """
        if wrong_level == sample.threat_level:
            return None

        prompt = (
            f"<|system|>\n{self.SYSTEM_PROMPT}\n"
            f"<|user|>\n{self._context_to_user_message(sample)}\n"
            "<|assistant|>\n"
        )

        chosen = self._build_assistant_response(sample)

        # Build plausible wrong response
        wrong_actions = random.choice(RECOMMENDED_ACTIONS[wrong_level])
        wrong_score = {"LOW": 0.1, "MEDIUM": 0.4, "HIGH": 0.6, "CRITICAL": 0.9}[wrong_level]
        wrong_resp = {
            "threat_level": wrong_level,
            "threat_score": wrong_score,
            "reasoning": f"Limited threat indicators observed. Level assessed as {wrong_level}.",
            "recommended_action": wrong_actions,
            "confidence": "LOW",
        }
        rejected = json.dumps(wrong_resp, indent=2)

        return DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)

    def format_all_sft(self, samples: List[ThreatSample]) -> List[Dict]:
        return [self.to_sft(s) for s in samples]

    def format_all_dpo(self, samples: List[ThreatSample]) -> List[DPOPair]:
        """Generate DPO pairs — one per sample, with a random wrong level"""
        pairs = []
        for s in samples:
            other_levels = [lvl for lvl in THREAT_LEVELS if lvl != s.threat_level]
            wrong_level = random.choice(other_levels)
            pair = self.to_dpo(s, wrong_level)
            if pair:
                pairs.append(pair)
        return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

class QualityFilter:
    """
    Removes samples that are:
    - Duplicates (by sample_id hash)
    - Inconsistent (score doesn't match level)
    - Too short in reasoning
    - Malformed JSON responses
    """

    MIN_REASONING_LEN = 30

    def __init__(self):
        self.seen_ids = set()
        self.stats = Counter()

    def _check_score_consistency(self, sample: Dict) -> bool:
        """Check threat score matches threat level"""
        level = sample["metadata"]["threat_level"]
        score = sample["metadata"]["threat_score"]
        ranges = {
            "LOW":      (0.00, 0.30),
            "MEDIUM":   (0.25, 0.55),  # slight overlap is OK
            "HIGH":     (0.45, 0.80),
            "CRITICAL": (0.70, 1.01),
        }
        lo, hi = ranges.get(level, (0, 1))
        return lo <= score <= hi

    def _check_reasoning_length(self, sample: Dict) -> bool:
        """Assistant message reasoning should be meaningful"""
        for msg in sample.get("messages", []):
            if msg["role"] == "assistant":
                try:
                    content = json.loads(msg["content"])
                    return len(content.get("reasoning", "")) >= self.MIN_REASONING_LEN
                except (json.JSONDecodeError, ValueError):
                    return False
        return False

    def _check_valid_json(self, sample: Dict) -> bool:
        """Assistant response must be valid JSON"""
        for msg in sample.get("messages", []):
            if msg["role"] == "assistant":
                try:
                    parsed = json.loads(msg["content"])
                    required = {"threat_level", "threat_score", "reasoning", "recommended_action"}
                    return required.issubset(parsed.keys())
                except (json.JSONDecodeError, ValueError):
                    return False
        return True

    def filter(self, samples: List[Dict]) -> List[Dict]:
        """Run all quality checks, return clean samples"""
        clean = []
        for s in samples:
            sid = s.get("sample_id", "")

            # Deduplication
            if sid in self.seen_ids:
                self.stats["duplicate"] += 1
                continue
            self.seen_ids.add(sid)

            # Score consistency
            if not self._check_score_consistency(s):
                self.stats["score_inconsistent"] += 1
                continue

            # Valid JSON response
            if not self._check_valid_json(s):
                self.stats["invalid_json"] += 1
                continue

            # Reasoning length
            if not self._check_reasoning_length(s):
                self.stats["short_reasoning"] += 1
                continue

            clean.append(s)
            self.stats["passed"] += 1

        total = len(samples)
        removed = total - len(clean)
        log.info("Quality filter: %d/%d passed (removed %d)", len(clean), total, removed)
        if removed > 0:
            log.info("  Rejection breakdown: %s", dict(self.stats))
        return clean


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset Splitter
# ─────────────────────────────────────────────────────────────────────────────

class DatasetSplitter:
    """
    Splits dataset into train/val/test with stratified sampling
    so each split has proportional threat level distribution.
    """

    def __init__(self, train: float = 0.80, val: float = 0.10, test: float = 0.10):
        assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
        self.train_ratio = train
        self.val_ratio   = val
        self.test_ratio  = test

    def split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Stratified split by threat_level"""
        # Group by threat level
        by_level: Dict[str, List[Dict]] = {lvl: [] for lvl in THREAT_LEVELS}
        for s in samples:
            lvl = s["metadata"]["threat_level"]
            by_level[lvl].append(s)

        train, val, test = [], [], []

        for lvl, group in by_level.items():
            random.shuffle(group)
            n = len(group)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)

            train.extend(group[:n_train])
            val.extend(group[n_train:n_train + n_val])
            test.extend(group[n_train + n_val:])

        # Final shuffle within each split
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
        return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset Exporter
# ─────────────────────────────────────────────────────────────────────────────

class DatasetExporter:
    """Exports datasets to JSONL (SFT) and JSONL (DPO) formats"""

    @staticmethod
    def save_jsonl(samples: List[Dict], path: Path):
        """Save list of dicts to a JSONL file"""
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        log.info("Saved %d samples → %s", len(samples), path)

    @staticmethod
    def save_dpo_jsonl(pairs: List[DPOPair], path: Path):
        """Save DPO pairs to JSONL"""
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({
                    "sample_id": p.sample_id,
                    "prompt": p.prompt,
                    "chosen": p.chosen,
                    "rejected": p.rejected,
                }, ensure_ascii=False) + "\n")
        log.info("Saved %d DPO pairs → %s", len(pairs), path)

    @staticmethod
    def save_raw_json(samples: List[ThreatSample], path: Path):
        """Save raw ThreatSample objects to JSON for inspection"""
        data = [s.to_dict() for s in samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Saved %d raw samples → %s", len(data), path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Statistics & Reporting
# ─────────────────────────────────────────────────────────────────────────────

class DatasetAnalyzer:
    """Generates statistics and quality reports for the dataset"""

    @staticmethod
    def analyze(train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict:
        all_splits = {"train": train, "val": val, "test": test}
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_samples": len(train) + len(val) + len(test),
            "splits": {},
        }

        for split_name, samples in all_splits.items():
            levels = Counter(s["metadata"]["threat_level"] for s in samples)
            scores = [s["metadata"]["threat_score"] for s in samples]

            report["splits"][split_name] = {
                "count": len(samples),
                "threat_level_distribution": dict(levels),
                "threat_score": {
                    "mean":  round(float(np.mean(scores)), 4) if scores else 0.0,
                    "std":   round(float(np.std(scores)), 4)  if scores else 0.0,
                    "min":   round(float(np.min(scores)), 4)  if scores else 0.0,
                    "max":   round(float(np.max(scores)), 4)  if scores else 0.0,
                },
            }

        # Token length estimates (rough: 1 token ≈ 4 chars)
        all_samples = train + val + test
        msg_lengths = []
        for s in all_samples:
            total_chars = sum(len(m["content"]) for m in s.get("messages", []))
            msg_lengths.append(total_chars // 4)

        if msg_lengths:
            report["token_length_estimates"] = {
                "mean_tokens":   int(np.mean(msg_lengths)),
                "max_tokens":    int(np.max(msg_lengths)),
                "min_tokens":    int(np.min(msg_lengths)),
                "p95_tokens":    int(np.percentile(msg_lengths, 95)),
            }

        return report

    @staticmethod
    def print_report(report: Dict):
        print("\n" + "=" * 60)
        print("  ASTAS DATASET REPORT")
        print("=" * 60)
        print(f"  Total Samples : {report['total_samples']}")
        print(f"  Generated At  : {report['generated_at']}")

        for split, info in report.get("splits", {}).items():
            print(f"\n  [{split.upper()}] {info['count']} samples")
            dist = info.get("threat_level_distribution", {})
            for lvl in THREAT_LEVELS:
                count = dist.get(lvl, 0)
                pct = count / info['count'] * 100 if info['count'] else 0
                bar = "█" * int(pct / 4)
                print(f"    {lvl:8s}: {count:5d}  {bar:<25s} {pct:.1f}%")
            sc = info.get("threat_score", {})
            print(f"    Score  mean={sc.get('mean', 0):.3f}  std={sc.get('std', 0):.3f}  "
                  f"min={sc.get('min', 0):.3f}  max={sc.get('max', 0):.3f}")

        tkn = report.get("token_length_estimates", {})
        if tkn:
            print(f"\n  Token Estimates (approx)")
            print(f"    Mean : {tkn.get('mean_tokens', 0)}")
            print(f"    P95  : {tkn.get('p95_tokens', 0)}")
            print(f"    Max  : {tkn.get('max_tokens', 0)}")

        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Orchestrates the full data preparation pipeline:
    generate → format → filter → split → export → report
    """

    def __init__(self, n_samples: int = 2000, seed: int = 42, balance: bool = True):
        self.n_samples = n_samples
        self.seed      = seed
        self.balance   = balance

        self.generator = RawSampleGenerator(seed=seed)
        self.formatter = PromptFormatter()
        self.filt      = QualityFilter()
        self.splitter  = DatasetSplitter(train=0.80, val=0.10, test=0.10)
        self.exporter  = DatasetExporter()
        self.analyzer  = DatasetAnalyzer()

    def run_sft(self):
        """Run supervised fine-tuning data pipeline"""
        log.info("=" * 60)
        log.info("ASTAS Data Prep — SFT Pipeline")
        log.info("=" * 60)
        log.info("Target samples: %d", self.n_samples)

        # 1. Generate raw samples
        log.info("\n[1/5] Generating raw samples...")
        raw_samples = self.generator.generate(self.n_samples, balance=self.balance)

        # Save raw for inspection
        self.exporter.save_raw_json(raw_samples, RAW_DIR / "raw_samples.json")

        # 2. Format to SFT chat format
        log.info("\n[2/5] Formatting to SFT chat format...")
        formatted = self.formatter.format_all_sft(raw_samples)

        # 3. Quality filter
        log.info("\n[3/5] Applying quality filters...")
        clean = self.filt.filter(formatted)

        # 4. Split
        log.info("\n[4/5] Splitting dataset...")
        train, val, test = self.splitter.split(clean)

        # 5. Export
        log.info("\n[5/5] Exporting...")
        self.exporter.save_jsonl(train, PROC_DIR / "train.jsonl")
        self.exporter.save_jsonl(val,   PROC_DIR / "val.jsonl")
        self.exporter.save_jsonl(test,  PROC_DIR / "test.jsonl")

        # Report
        report = self.analyzer.analyze(train, val, test)
        with open(STATS_DIR / "dataset_report.json", "w") as f:
            json.dump(report, f, indent=2)
        self.analyzer.print_report(report)

        log.info("✓ SFT pipeline complete")
        return train, val, test

    def run_dpo(self):
        """Run DPO preference dataset pipeline"""
        log.info("=" * 60)
        log.info("ASTAS Data Prep — DPO Pipeline")
        log.info("=" * 60)

        # Generate HIGH/CRITICAL samples only for DPO
        # (preference learning most valuable on edge cases)
        log.info("\n[1/4] Generating DPO source samples...")
        raw_samples = self.generator.generate(self.n_samples, balance=True)

        # 2. Build DPO pairs
        log.info("\n[2/4] Building preference pairs...")
        dpo_pairs = self.formatter.format_all_dpo(raw_samples)
        log.info("  Built %d DPO pairs", len(dpo_pairs))

        # 3. Split 80/20 train/val for DPO
        random.shuffle(dpo_pairs)
        split_idx = int(len(dpo_pairs) * 0.8)
        dpo_train = dpo_pairs[:split_idx]
        dpo_val   = dpo_pairs[split_idx:]

        # 4. Export
        log.info("\n[3/4] Exporting DPO pairs...")
        self.exporter.save_dpo_jsonl(dpo_train, DPO_DIR / "train_dpo.jsonl")
        self.exporter.save_dpo_jsonl(dpo_val,   DPO_DIR / "val_dpo.jsonl")

        log.info("✓ DPO pipeline complete: %d train, %d val", len(dpo_train), len(dpo_val))
        return dpo_train, dpo_val

    def run_full(self):
        """Run both SFT and DPO pipelines"""
        sft_results = self.run_sft()
        dpo_results = self.run_dpo()
        return sft_results, dpo_results

    def validate(self):
        """Validate existing dataset files"""
        log.info("Validating existing dataset...")
        files = {
            "train": PROC_DIR / "train.jsonl",
            "val":   PROC_DIR / "val.jsonl",
            "test":  PROC_DIR / "test.jsonl",
        }
        results = {}
        for split, path in files.items():
            if not path.exists():
                log.warning("  MISSING: %s", path)
                results[split] = {"status": "missing"}
                continue

            samples = []
            errors = 0
            with open(path) as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        errors += 1

            results[split] = {
                "status": "ok" if errors == 0 else "errors",
                "count": len(samples),
                "parse_errors": errors,
                "threat_levels": dict(Counter(
                    s["metadata"]["threat_level"] for s in samples
                ))
            }
            log.info("  %s: %d samples, %d parse errors", split, len(samples), errors)

        return results

    def show_sample(self, n: int = 2):
        """Show n sample formatted examples"""
        raw = self.generator.generate(n, balance=False)
        for i, sample in enumerate(raw):
            formatted = self.formatter.to_sft(sample)
            print(f"\n{'='*60}")
            print(f"  SAMPLE {i+1}: {sample.threat_level} (score={sample.threat_score:.3f})")
            print(f"{'='*60}")
            for msg in formatted["messages"]:
                role = msg["role"].upper()
                content = msg["content"]
                if role == "SYSTEM":
                    print(f"[{role}] (omitted for brevity)")
                else:
                    print(f"\n[{role}]\n{content}")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASTAS LLM Finetuning Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_prep.py                       # Full SFT pipeline, 2000 samples
  python data_prep.py --samples 5000        # 5000 samples
  python data_prep.py --mode dpo            # DPO preference dataset
  python data_prep.py --mode full           # Both SFT + DPO
  python data_prep.py --validate            # Validate existing dataset
  python data_prep.py --stats               # Show dataset statistics
  python data_prep.py --preview             # Preview 2 formatted samples
  python data_prep.py --no-balance          # Skew distribution (realistic)
        """
    )
    parser.add_argument("--samples",    type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--mode",       choices=["sft", "dpo", "full"], default="sft")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-balance", action="store_true", help="Use natural (skewed) class distribution")
    parser.add_argument("--validate",   action="store_true", help="Validate existing dataset")
    parser.add_argument("--stats",      action="store_true", help="Show dataset stats")
    parser.add_argument("--preview",    action="store_true", help="Preview formatted samples")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = DataPipeline(
        n_samples=args.samples,
        seed=args.seed,
        balance=not args.no_balance
    )

    if args.validate:
        results = pipeline.validate()
        print(json.dumps(results, indent=2))
        return

    if args.stats:
        files = {"train": PROC_DIR / "train.jsonl",
                 "val":   PROC_DIR / "val.jsonl",
                 "test":  PROC_DIR / "test.jsonl"}
        splits = {}
        for name, path in files.items():
            if path.exists():
                with open(path) as f:
                    splits[name] = [json.loads(l) for l in f]
        if splits:
            report = DatasetAnalyzer.analyze(
                splits.get("train", []),
                splits.get("val", []),
                splits.get("test", [])
            )
            DatasetAnalyzer.print_report(report)
        else:
            log.warning("No dataset found. Run without --stats first.")
        return

    if args.preview:
        pipeline.show_sample(n=2)
        return

    if args.mode == "sft":
        pipeline.run_sft()
    elif args.mode == "dpo":
        pipeline.run_dpo()
    else:
        pipeline.run_full()


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Aliases — these names are imported by main.py
# ─────────────────────────────────────────────────────────────────────────────

# main.py calls:
#   from LLM_Enigne.Finetunning_Model.data_prep import ThreatScenarioGenerator, SFTFormatter, DataSplitter

ThreatScenarioGenerator = RawSampleGenerator   # same class, expected name
SFTFormatter            = PromptFormatter      # same class, expected name
DataSplitter            = DatasetSplitter      # same class, expected name