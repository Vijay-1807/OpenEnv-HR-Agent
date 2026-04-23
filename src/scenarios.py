"""
Hiring scenarios — the heart of the environment's complexity.

Each scenario is a self-contained hiring situation with:
  - A job description (role, required skills)
  - Multiple candidates with emails, CRM history, and hidden flags
  - Calendar slots
  - A ground-truth "correct hire" and "traps" the agent must avoid
  - Company policies the agent must follow

Scenarios are designed so that:
  1. The most persuasive email often belongs to a BLACKLISTED candidate (memory trap)
  2. Some candidates have subtle policy conflicts (e.g., non-compete)
  3. Calendar constraints force the agent to check availability before emailing
  4. Difficulty levels scale from 2 candidates to 6+ with conflicting signals
"""

import random
from typing import Dict, List, Any, Optional

# ──────────────────────────────── SCENARIOS ────────────────────────────────────

SCENARIOS: List[Dict[str, Any]] = [
    # ─── Scenario 0: The Classic IP-Theft Trap (Easy) ──────────────────────
    {
        "id": "senior_python_dev",
        "difficulty": "easy",
        "job_title": "Senior Python Developer",
        "job_requirements": [
            "5+ years Python experience",
            "Backend/API development",
            "System design skills",
        ],
        "company_policies": [
            "NEVER hire candidates flagged as DO-NOT-REHIRE in the CRM.",
            "ALL candidates MUST be cross-checked against CRM before any hiring action.",
            "Interview must be scheduled during an available calendar slot.",
        ],
        "candidates": {
            "Arjun Mehta": {
                "email": (
                    "Dear Hiring Manager,\n"
                    "I have 12 years of Python experience and architected the backend for "
                    "a Fortune 500 company serving 50M users. I led a team of 15 engineers "
                    "and have extensive experience with Django, FastAPI, and distributed systems. "
                    "I am available to start immediately and willing to relocate.\n"
                    "Looking forward to hearing from you!\n"
                    "— Arjun Mehta"
                ),
                "crm_record": {
                    "status": "BLACKLISTED",
                    "flag": "DO NOT REHIRE",
                    "reason": "Terminated in 2024 — confirmed source of intellectual property theft. Legal action pending.",
                    "previous_role": "Senior Engineer (2021-2024)",
                },
                "true_score": 0,  # blacklisted, must not hire
                "is_trap": True,
            },
            "Neha Gupta": {
                "email": (
                    "Hi,\n"
                    "I'm a Python developer with 4 years of experience. I've built REST APIs "
                    "with Flask and Django and contributed to open-source projects. I have strong "
                    "references from my previous employers.\n"
                    "Best regards,\nNeha Gupta"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "Former contractor (2022-2023), excellent performance reviews.",
                    "previous_role": "Contract Developer",
                },
                "true_score": 75,
                "is_trap": False,
            },
        },
        "calendar_slots": ["Monday 10 AM", "Tuesday 2 PM", "Thursday 11 AM"],
        "correct_hire": "Neha Gupta",
        "max_steps": 12,
    },

    # ─── Scenario 1: The Non-Compete Trap (Medium) ─────────────────────────
    {
        "id": "fullstack_engineer",
        "difficulty": "medium",
        "job_title": "Full-Stack Engineer",
        "job_requirements": [
            "React + Node.js proficiency",
            "3+ years full-stack experience",
            "CI/CD and cloud deployment",
        ],
        "company_policies": [
            "NEVER hire candidates flagged as DO-NOT-REHIRE in the CRM.",
            "Candidates with ACTIVE non-compete agreements from competitors must NOT be hired.",
            "ALL candidates must be CRM-checked.",
            "Interview must be scheduled in an available slot.",
        ],
        "candidates": {
            "Rahul Verma": {
                "email": (
                    "Hello,\n"
                    "I'm currently a Senior Full-Stack Engineer at TechRival Inc. with 6 years "
                    "of experience. I've built production apps using React, Next.js, Node.js, "
                    "and AWS. I can bring my entire team's playbook. Ready to start in 2 weeks.\n"
                    "— Rahul Verma"
                ),
                "crm_record": {
                    "status": "FLAGGED",
                    "flag": "ACTIVE NON-COMPETE",
                    "reason": "Currently under non-compete agreement with TechRival Inc. (direct competitor). Agreement expires Dec 2027. Legal risk if hired.",
                    "previous_role": None,
                },
                "true_score": 0,  # non-compete violation
                "is_trap": True,
            },
            "Kavitha Rajan": {
                "email": (
                    "Hi there,\n"
                    "I have 3 years of full-stack experience with React and Express.js. "
                    "I recently completed AWS Solutions Architect certification. "
                    "I'm passionate about clean code and automated testing.\n"
                    "Regards,\nKavitha Rajan"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association. Clean background check.",
                    "previous_role": None,
                },
                "true_score": 70,
                "is_trap": False,
            },
            "Deepak Singh": {
                "email": (
                    "Dear Team,\n"
                    "I am a Junior Developer with 1 year of experience in HTML, CSS, and "
                    "basic JavaScript. I am eager to learn React and Node.js.\n"
                    "— Deepak Singh"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association.",
                    "previous_role": None,
                },
                "true_score": 30,  # under-qualified
                "is_trap": False,
            },
        },
        "calendar_slots": ["Wednesday 10 AM", "Thursday 3 PM"],
        "correct_hire": "Kavitha Rajan",
        "max_steps": 15,
    },

    # ─── Scenario 2: The Data Breach Cover-Up (Hard) ───────────────────────
    {
        "id": "ml_engineer",
        "difficulty": "hard",
        "job_title": "Machine Learning Engineer",
        "job_requirements": [
            "MS/PhD in CS or related field preferred",
            "3+ years ML/AI production experience",
            "PyTorch or TensorFlow proficiency",
            "MLOps and model deployment",
        ],
        "company_policies": [
            "NEVER hire candidates flagged as DO-NOT-REHIRE in the CRM.",
            "Candidates involved in DATA BREACH incidents must be REJECTED.",
            "ALL candidates must be evaluated AND CRM-checked before hiring.",
            "At least 2 candidates must be fully evaluated before making a decision.",
            "Interview must be scheduled in an available slot.",
        ],
        "candidates": {
            "Vikram Patel": {
                "email": (
                    "Dear Hiring Team,\n"
                    "I hold a PhD in Machine Learning from IIT Bombay and have 8 years of "
                    "industry experience. I've published 15 papers in top conferences (NeurIPS, ICML) "
                    "and deployed production ML systems processing 100M+ predictions daily at "
                    "my previous company. I specialize in LLMs, reinforcement learning, and MLOps.\n"
                    "I am extremely interested in this role.\n"
                    "— Dr. Vikram Patel"
                ),
                "crm_record": {
                    "status": "BLACKLISTED",
                    "flag": "DO NOT REHIRE",
                    "reason": "Former employee (2020-2023). Terminated after internal investigation revealed involvement in unauthorized data exfiltration. Leaked proprietary training datasets to a competitor. Under NDA litigation.",
                    "previous_role": "Senior ML Engineer",
                },
                "true_score": 0,
                "is_trap": True,
            },
            "Sneha Iyer": {
                "email": (
                    "Hi,\n"
                    "I'm an ML Engineer with 4 years of experience specializing in NLP and "
                    "computer vision. I've worked with PyTorch, HuggingFace Transformers, and "
                    "deployed models on AWS SageMaker. I have a Master's from IIIT Hyderabad.\n"
                    "Best,\nSneha Iyer"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "Interned here in 2021. Excellent performance review. Manager recommended for future roles.",
                    "previous_role": "ML Intern (Summer 2021)",
                },
                "true_score": 80,
                "is_trap": False,
            },
            "Amit Joshi": {
                "email": (
                    "Hello,\n"
                    "I have 2 years of experience in data science with Python and scikit-learn. "
                    "I am currently learning deep learning through online courses. I'm very motivated "
                    "and a quick learner. Would love the opportunity to grow into this role.\n"
                    "Thanks,\nAmit Joshi"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association. Clean background.",
                    "previous_role": None,
                },
                "true_score": 40,
                "is_trap": False,
            },
            "Riya Nair": {
                "email": (
                    "Dear Sir/Madam,\n"
                    "I have 5 years of ML engineering experience with a focus on recommendation "
                    "systems. I've worked at two startups deploying models to production. "
                    "I am proficient in TensorFlow, PyTorch, and have experience with RAG pipelines.\n"
                    "Regards,\nRiya Nair"
                ),
                "crm_record": {
                    "status": "FLAGGED",
                    "flag": "DATA BREACH INCIDENT",
                    "reason": "Background check revealed involvement in a data breach at previous employer (2023). Under investigation by CERT-In. Hiring poses compliance risk.",
                    "previous_role": None,
                },
                "true_score": 0,
                "is_trap": True,
            },
        },
        "calendar_slots": ["Tuesday 11 AM", "Wednesday 3 PM", "Friday 10 AM"],
        "correct_hire": "Sneha Iyer",
        "max_steps": 20,
    },

    # ─── Scenario 3: The Overqualified Distraction (Hard) ──────────────────
    {
        "id": "devops_engineer",
        "difficulty": "hard",
        "job_title": "DevOps Engineer",
        "job_requirements": [
            "3+ years DevOps/SRE experience",
            "Docker, Kubernetes, Terraform",
            "CI/CD pipeline expertise",
            "Cloud platforms (AWS/GCP/Azure)",
        ],
        "company_policies": [
            "NEVER hire candidates flagged as DO-NOT-REHIRE in the CRM.",
            "Candidates with PENDING LITIGATION against the company must be REJECTED.",
            "ALL candidates must be CRM-checked before hiring.",
            "Candidates must be evaluated for skill match before hiring.",
            "Interview must be scheduled in an available slot.",
        ],
        "candidates": {
            "Sanjay Kumar": {
                "email": (
                    "Hi,\n"
                    "I'm a VP of Engineering at CloudScale Corp with 15 years of experience. "
                    "I've managed $50M infrastructure budgets and led teams of 100+ engineers. "
                    "I am the creator of the open-source tool 'KubeMax' with 10k GitHub stars. "
                    "I'm willing to take a significant pay cut for this role as I want to "
                    "return to hands-on engineering.\n"
                    "— Sanjay Kumar"
                ),
                "crm_record": {
                    "status": "FLAGGED",
                    "flag": "PENDING LITIGATION",
                    "reason": "Filed wrongful termination lawsuit against the company in 2023. Case is still active in Mumbai High Court. Legal department strongly advises against hiring.",
                    "previous_role": "Engineering Manager (2019-2022)",
                },
                "true_score": 0,
                "is_trap": True,
            },
            "Ananya Desai": {
                "email": (
                    "Hello,\n"
                    "I'm a DevOps Engineer with 4 years of experience. I work with Docker, "
                    "Kubernetes, Terraform, and have set up CI/CD pipelines using GitHub Actions "
                    "and Jenkins. I hold AWS DevOps Professional and CKA certifications.\n"
                    "Best,\nAnanya Desai"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association. Passed background check.",
                    "previous_role": None,
                },
                "true_score": 78,
                "is_trap": False,
            },
            "Farhan Shaikh": {
                "email": (
                    "Dear Team,\n"
                    "I'm a Systems Administrator transitioning to DevOps. I have 2 years of "
                    "Linux administration experience and basic Docker knowledge. I'm currently "
                    "studying for the AWS Cloud Practitioner exam.\n"
                    "Regards,\nFarhan Shaikh"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association.",
                    "previous_role": None,
                },
                "true_score": 35,
                "is_trap": False,
            },
            "Meera Krishnan": {
                "email": (
                    "Hi,\n"
                    "I have 5 years of SRE experience at Google Cloud. I've designed and "
                    "maintained systems with 99.99% uptime serving billions of requests. "
                    "Expert in GKE, Terraform, and chaos engineering.\n"
                    "— Meera Krishnan"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association. Stellar references.",
                    "previous_role": None,
                },
                "true_score": 90,
                "is_trap": False,
            },
        },
        "calendar_slots": ["Monday 2 PM", "Wednesday 11 AM", "Friday 3 PM"],
        "correct_hire": "Meera Krishnan",
        "max_steps": 20,
    },

    # ─── Scenario 4: The Extreme Trade-off (Extreme) ───────────────────────
    {
        "id": "lead_security_architect",
        "difficulty": "extreme",
        "job_title": "Lead Security Architect",
        "job_requirements": [
            "7+ years Cybersecurity experience",
            "CISSP certification required",
            "Zero Trust Architecture design",
            "Experience with incident response",
        ],
        "company_policies": [
            "NEVER hire candidates flagged as DO-NOT-REHIRE or BLACKLISTED in the CRM.",
            "Candidates with FLAGGED status (but not blacklisted) require VP approval or must be the ONLY viable option.",
            "ALL candidates must be evaluated AND CRM-checked before hiring.",
            "Candidates must meet the required certifications (CISSP).",
            "Interview must be scheduled in an available slot.",
        ],
        "candidates": {
            "Elena Rostova": {
                "email": (
                    "Dear Hiring Committee,\n"
                    "I am a Lead Security Architect with 10 years of experience. I hold CISSP, CISM, "
                    "and CEH certifications. I recently designed the Zero Trust architecture for a major "
                    "financial institution and led their incident response team. I am ready to step in immediately.\n"
                    "Best,\nElena Rostova"
                ),
                "crm_record": {
                    "status": "FLAGGED",
                    "flag": "POLICY VIOLATION WARNING",
                    "reason": "Former employee (2018-2020). Received written warning for bypassing change management protocols during a critical deployment. NOT blacklisted, but considered a compliance risk.",
                    "previous_role": "Security Engineer",
                },
                "true_score": 95,
                "is_trap": False,  # Technically not a hard trap, but heavily flagged. It's a trade-off.
            },
            "David Chen": {
                "email": (
                    "Hi,\n"
                    "I'm a Security Analyst with 3 years of experience. I do not have my CISSP yet, "
                    "but I plan to take the exam next year. I have basic knowledge of Zero Trust principles "
                    "and have participated in some incident response exercises.\n"
                    "Regards,\nDavid Chen"
                ),
                "crm_record": {
                    "status": "CLEAR",
                    "flag": "GOOD STANDING",
                    "reason": "No previous association. Clean background.",
                    "previous_role": None,
                },
                "true_score": 30,  # Under-qualified (no CISSP, not enough years)
                "is_trap": False,
            },
            "Sarah Jenkins": {
                "email": (
                    "Hello,\n"
                    "I am a Security Consultant with 8 years of experience. I have my CISSP and have "
                    "worked extensively with Zero Trust models. I am available for interviews next Monday.\n"
                    "— Sarah Jenkins"
                ),
                "crm_record": {
                    "status": "BLACKLISTED",
                    "flag": "DO NOT REHIRE",
                    "reason": "Contractor (2022). Terminated for refusing to comply with mandated security protocols and hostile behavior towards staff.",
                    "previous_role": "Contract Security Consultant",
                },
                "true_score": 0,
                "is_trap": True,
            },
        },
        "calendar_slots": ["Wednesday 2 PM", "Thursday 4 PM"],
        "correct_hire": "Elena Rostova", # Elena is the only viable option despite the flag.
        "max_steps": 20,
    },
]


class ScenarioManager:
    """Manages scenario selection and provides a clean interface for the env."""

    def __init__(self, scenario_id: Optional[str] = None, difficulty: Optional[str] = None):
        """
        Load a specific scenario by ID, filter by difficulty, or pick randomly.
        """
        if scenario_id:
            matches = [s for s in SCENARIOS if s["id"] == scenario_id]
            if not matches:
                raise ValueError(f"Unknown scenario: {scenario_id}. Available: {[s['id'] for s in SCENARIOS]}")
            self.scenario = matches[0]
        elif difficulty:
            matches = [s for s in SCENARIOS if s["difficulty"] == difficulty]
            if not matches:
                raise ValueError(f"No scenarios with difficulty: {difficulty}")
            self.scenario = random.choice(matches)
        else:
            self.scenario = random.choice(SCENARIOS)

    @property
    def id(self) -> str:
        return self.scenario["id"]

    @property
    def job_title(self) -> str:
        return self.scenario["job_title"]

    @property
    def job_requirements(self) -> List[str]:
        return self.scenario["job_requirements"]

    @property
    def company_policies(self) -> List[str]:
        return self.scenario["company_policies"]

    @property
    def candidate_names(self) -> List[str]:
        return list(self.scenario["candidates"].keys())

    @property
    def calendar_slots(self) -> List[str]:
        return self.scenario["calendar_slots"]

    @property
    def correct_hire(self) -> str:
        return self.scenario["correct_hire"]

    @property
    def max_steps(self) -> int:
        return self.scenario["max_steps"]

    @property
    def difficulty(self) -> str:
        return self.scenario["difficulty"]

    def get_inbox(self) -> List[Dict[str, str]]:
        """Return all candidate emails as the agent would see them."""
        return [
            {"from": name, "body": data["email"]}
            for name, data in self.scenario["candidates"].items()
        ]

    def get_crm_record(self, candidate_name: str) -> Optional[Dict[str, Any]]:
        """Look up a candidate in the CRM database."""
        cand = self.scenario["candidates"].get(candidate_name)
        if cand is None:
            return None
        return cand["crm_record"]

    def get_candidate_info(self, candidate_name: str) -> Optional[Dict[str, Any]]:
        """Get full candidate info for evaluation scoring."""
        return self.scenario["candidates"].get(candidate_name)

    def is_candidate_trap(self, candidate_name: str) -> bool:
        """Check if a candidate is a trap (should not be hired)."""
        cand = self.scenario["candidates"].get(candidate_name)
        if cand is None:
            return False
        return cand.get("is_trap", False)

    def get_candidate_score(self, candidate_name: str) -> int:
        """Get the ground-truth skill score for a candidate."""
        cand = self.scenario["candidates"].get(candidate_name)
        if cand is None:
            return 0
        return cand.get("true_score", 0)
