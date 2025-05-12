from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import json
from datetime import datetime
import secrets
from functools import wraps
import threading
import re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ------ADDED------ Sentry initialization
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],                 
    integrations=[FlaskIntegration()],             
    traces_sample_rate=0.5,                    
    environment=os.environ.get("FLASK_ENV", "production")
)



# ------ADDED------ Structured JSON logging
from logging.handlers import RotatingFileHandler
class JSONFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "module":    record.module,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

# Configure handler with JSONFormatter
def configure_logging():
    handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

configure_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Use a strong secret key with proper length (at least 32 bytes)
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
# Configure session to be more secure
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('PRODUCTION', 'False').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Define available models with display names and local paths
MODELS = {
    "Telugu BERT": r"E:/Ashish NLP Project/results/telugu-bert-finetuned/final-model",
    "XLM-RoBERTa": r"E:/NLP_NEW_MODELS/NLP_FACE/FacebookAI/xlm-roberta-base/final-model",
    "IndicBERT": r"E:/NLP_NEW_MODELS/IndicBERT-bert-finetuned/final-model",
    "MuRIL": r"E:/NLP_NEW_MODELS/NLP_MuRIL/google/muril-base-cased/final-model"
}



# Load models and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}
models_loaded = False
loading_lock = threading.Lock()


# ------ADDED------ Prometheus metrics initialization
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app, path="/metrics")
metrics.info(
    'app_info',
    'Application info',
    version='1.0.0',
    models=len(MODELS)
)


# Mental health resources by condition
RESOURCES = {
    "Anxiety": {
        "description": "Anxiety disorders are characterized by persistent and excessive worry about various situations, affecting daily activities and often producing physical symptoms like rapid heartbeat, shortness of breath, and restlessness.",
        "self_help": [
            "Practice deep breathing exercises (4-4-8 technique: inhale for 4, hold for 4, exhale for 8)",
            "Try progressive muscle relaxation by tensing and relaxing each muscle group",
            "Maintain a regular sleep schedule with 7-8 hours of sleep",
            "Limit caffeine, tea, and processed foods",
            "Exercise for at least 30 minutes daily, even if just walking",
            "Use mindfulness meditation to stay present and reduce anticipatory anxiety",
            "Create a worry journal to document and challenge anxious thoughts",
            "Practice grounding techniques like the 5-4-3-2-1 method",
            "Try yoga and pranayama (breathing exercises) which are traditional Indian practices for mental wellbeing"
        ],
        "therapies": [
            "Cognitive Behavioral Therapy (CBT) - helps identify and change negative thought patterns",
            "Exposure Therapy - gradual exposure to anxiety triggers in a safe environment",
            "Acceptance and Commitment Therapy (ACT) - focuses on accepting anxiety while still taking action",
            "Mindfulness-Based Stress Reduction (MBSR) - combines mindfulness and yoga",
            "Art Therapy - expresses emotions through creative means",
            "Ayurvedic counseling - traditional Indian approach that considers mind-body connection"
        ],
        "apps": [
            {"name": "Wysa", "url": "https://www.wysa.io", "description": "Indian AI chatbot for emotional wellness and anxiety support"},
            {"name": "InnerHour", "url": "https://www.theinnerhour.com", "description": "Indian self-help app with personalized mental health plans"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian mental wellness platform with expert guidance"},
            {"name": "Mindhouse", "url": "https://www.mindhouse.com", "description": "Indian meditation and yoga app focused on stress reduction"},
            {"name": "Let's Talk", "url": "https://letstalk.co.in", "description": "Indian platform connecting users with mental health professionals"}
        ],
        "helplines": [
            {"name": "NIMHANS Toll-Free Mental Health Helpline", "number": "080-46110007"},
            {"name": "iCALL", "number": "022-25521111", "hours": "Mon-Sat, 8 AM to 10 PM"},
            {"name": "Mann Talks", "number": "8686139139"}
        ]
    },
    "Normal": {
        "description": "Your text doesn't indicate significant mental health concerns at this time. Maintaining mental wellbeing through preventative practices is important for continued psychological health.",
        "self_help": [
            "Continue practicing good self-care with regular exercise, healthy diet, and adequate rest",
            "Maintain healthy social connections and nurture important relationships",
            "Engage in regular physical activity that you enjoy",
            "Practice mindfulness through meditation or yoga",
            "Get adequate sleep (7-9 hours) and maintain consistent sleep patterns",
            "Set realistic goals and celebrate achievements",
            "Engage in activities that provide meaning and purpose",
            "Practice gratitude by noting three positive experiences daily",
            "Explore traditional Indian wellness practices like yoga nidra and meditation"
        ],
        "therapies": [
            "Life coaching for personal growth and goal achievement",
            "Wellness counseling for maintaining balanced lifestyle",
            "Routine mental health check-ups to monitor wellbeing",
            "Positive psychology interventions to enhance strengths",
            "Mindfulness training for present moment awareness",
            "Preventative stress management education",
            "Traditional Indian approaches like yoga therapy"
        ],
        "apps": [
            {"name": "StepSetGo", "url": "https://www.stepsetgo.com", "description": "Indian fitness app that rewards physical activity"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian mental wellness platform with expert guidance"},
            {"name": "ThinkRight.me", "url": "https://thinkright.me", "description": "Indian meditation and mindfulness app with guided sessions in Hindi and English"},
            {"name": "Evolove", "url": "https://evolove.in", "description": "Indian holistic wellness app focusing on mind-body connection"},
            {"name": "Mindhouse", "url": "https://www.mindhouse.com", "description": "Indian meditation and yoga app for daily practice"}
        ]
    },
    "Depression": {
        "description": "Depression is a mood disorder causing persistent feelings of sadness, emptiness, and loss of interest that can affect how you feel, think, and handle daily activities. It may include physical symptoms like sleep changes, appetite changes, and fatigue.",
        "self_help": [
            "Set small, achievable daily goals and celebrate completing them",
            "Exercise regularly, even if just a 10-minute walk daily",
            "Maintain social connections even when you feel like isolating",
            "Follow a regular sleeping pattern by going to bed and waking at consistent times",
            "Challenge negative thoughts by identifying and questioning them",
            "Practice self-compassion by speaking to yourself kindly",
            "Spend time in nature and in sunlight when possible",
            "Try including traditional Indian spices like turmeric in your diet, which have mood-enhancing properties",
            "Practice yoga asanas that elevate mood like backbends and sun salutations",
            "Limit alcohol and avoid recreational drugs"
        ],
        "therapies": [
            "Cognitive Behavioral Therapy (CBT) - addresses negative thought patterns",
            "Interpersonal Therapy - focuses on relationship problems",
            "Behavioral Activation - increases engagement in positive activities",
            "Psychodynamic Therapy - explores unconscious conflicts",
            "Acceptance and Commitment Therapy (ACT) - builds psychological flexibility",
            "Mindfulness-Based Cognitive Therapy (MBCT) - combines mindfulness and CBT",
            "Traditional Indian approaches like Ayurvedic counseling"
        ],
        "apps": [
            {"name": "Wysa", "url": "https://www.wysa.io", "description": "Indian AI chatbot for emotional support"},
            {"name": "InnerHour", "url": "https://www.theinnerhour.com", "description": "Indian app with depression-specific programs"},
            {"name": "Evolove", "url": "https://evolove.in", "description": "Indian app focusing on mind-body wellness"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian platform connecting users with mental health experts"},
            {"name": "Juno Clinic", "url": "https://www.junoclinic.com", "description": "Indian teletherapy app with depression specialists"}
        ],
        "helplines": [
            {"name": "NIMHANS Depression Helpline", "number": "080-46110007"},
            {"name": "Vandrevala Foundation", "number": "1860-2662-345 / 1800-2333-330", "hours": "24x7"},
            {"name": "iCALL", "number": "022-25521111", "hours": "Mon-Sat, 8 AM to 10 PM"}
        ]
    },
    "Suicidal": {
        "description": "If you're having thoughts of suicide, please seek immediate help. You're not alone, and support is available. Suicidal thoughts are serious but can be managed with proper support.",
        "emergency": "CALL EMERGENCY SERVICES (102 or 112) OR GO TO YOUR NEAREST EMERGENCY ROOM IMMEDIATELY IF YOU ARE IN CRISIS",
        "self_help": [
            "Remove access to means of self-harm immediately",
            "Avoid alcohol and drugs which can intensify negative thoughts",
            "Create a detailed safety plan with contacts and resources",
            "Use distraction techniques when thoughts arise (music, physical activity, calling a friend)",
            "Practice grounding exercises using the 5-4-3-2-1 senses technique",
            "Implement hope box (physical or digital collection of positive memories)",
            "Use postponement strategy - agree to wait 24 hours before acting on thoughts"
        ],
        "therapies": [
            "Dialectical Behavior Therapy (DBT) - specifically effective for suicidal thoughts",
            "Cognitive Behavioral Therapy for Suicide Prevention (CBT-SP)",
            "Collaborative Assessment and Management of Suicidality (CAMS)",
            "Brief Cognitive Behavioral Therapy (BCBT) for suicide",
            "Attachment-Based Family Therapy for adolescents",
            "Problem-Solving Therapy - builds coping skills"
        ],
        "apps": [
            {"name": "JIVAN", "url": "https://jivanapp.org", "description": "Indian suicide prevention app with emergency contacts"},
            {"name": "Wysa", "url": "https://www.wysa.io", "description": "Indian AI chatbot with crisis resources"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian platform with 24/7 SOS support"},
            {"name": "Mann Talks", "url": "https://www.manntalks.org", "description": "Indian mental health support app with crisis intervention"}
        ],
        "helplines": [
            {"name": "AASRA", "number": "91-9820466726", "hours": "24x7"},
            {"name": "Vandrevala Foundation", "number": "1860-2662-345 / 1800-2333-330", "hours": "24x7"},
            {"name": "Sneha India", "number": "044-24640050", "hours": "24x7"},
            {"name": "Sumaitri", "number": "011-23389090", "hours": "2 PM to 10 PM"},
            {"name": "Cooj Mental Health Foundation", "number": "0832-2252525", "hours": "Mon-Fri, 1 PM to 7 PM"},
            {"name": "Sahai", "number": "080-25497777", "hours": "10 AM to 8 PM"},
            {"name": "Roshni Trust", "number": "040-66202000", "hours": "11 AM to 9 PM"},
            {"name": "Lifeline Foundation", "number": "033-24637401", "hours": "10 AM to 6 PM"}
        ]
    },
    "Stress": {
        "description": "Stress is your body's reaction to pressure from certain situations or events in your life, leading to physical and emotional responses. Chronic stress can contribute to numerous health problems if not managed effectively.",
        "self_help": [
            "Practice time management techniques using priority matrices and calendar blocking",
            "Engage in regular physical activity (30 minutes daily, 5 days a week)",
            "Use deep breathing exercises or pranayama several times daily",
            "Maintain proper nutrition with emphasis on whole foods and hydration (2-3 liters daily)",
            "Ensure adequate sleep (7-9 hours) with consistent bedtime routine",
            "Take regular breaks during work (5-10 minutes every hour)",
            "Set boundaries between work and personal life",
            "Practice saying no to additional commitments when overwhelmed",
            "Try traditional Indian stress management practices like yoga nidra or meditation",
            "Engage in enjoyable activities daily"
        ],
        "therapies": [
            "Stress management counseling with personalized coping strategies",
            "Mindfulness-Based Stress Reduction (MBSR) - 8-week structured program",
            "Biofeedback to gain awareness of physiological functions",
            "Relaxation training with guided progressive muscle relaxation",
            "Cognitive restructuring to identify and change stress-inducing thoughts",
            "Solution-focused brief therapy for specific stressors",
            "Yoga therapy - traditional Indian approach to stress management"
        ],
        "apps": [
            {"name": "NirvanaFitness", "url": "https://nirvanafitness.in", "description": "Indian breathing and meditation app"},
            {"name": "ThinkRight.me", "url": "https://thinkright.me", "description": "Indian meditation app with guided sessions in Hindi and English"},
            {"name": "Mindhouse", "url": "https://www.mindhouse.com", "description": "Indian yoga and meditation app"},
            {"name": "Evolove", "url": "https://evolove.in", "description": "Indian holistic wellness app"},
            {"name": "InnerHour", "url": "https://www.theinnerhour.com", "description": "Indian app with stress-specific programs"}
        ],
        "helplines": [
            {"name": "NIMHANS Toll-Free Mental Health Helpline", "number": "080-46110007"},
            {"name": "iCall", "number": "022-25521111", "hours": "Mon-Sat, 8 AM to 10 PM"},
            {"name": "Mann Talks", "number": "8686139139"}
        ]
    },
    "Bipolar": {
        "description": "Bipolar disorder causes unusual shifts in mood, energy, activity levels, concentration, and ability to carry out day-to-day tasks. It involves episodes of mania or hypomania (elevated mood) and depression.",
        "self_help": [
            "Maintain a consistent daily routine with regular sleep, meals, and activities",
            "Track your moods, sleep, and symptoms in a detailed mood journal",
            "Get regular sleep (7-9 hours) at consistent times",
            "Avoid alcohol and recreational drugs which can trigger episodes",
            "Exercise regularly (30 minutes, 5 days a week) to regulate mood",
            "Build a comprehensive support network including family, friends, and professionals",
            "Learn to identify early warning signs of mood episodes",
            "Create an action plan for managing early symptoms",
            "Practice stress management techniques daily",
            "Consider incorporating yoga and traditional Indian wellness practices into your routine"
        ],
        "therapies": [
            "Interpersonal and Social Rhythm Therapy (IPSRT) - stabilizes daily routines",
            "Cognitive Behavioral Therapy (CBT) - addresses negative thought patterns",
            "Family-Focused Therapy - improves family communication",
            "Psychoeducation about managing bipolar effectively",
            "Dialectical Behavior Therapy (DBT) - improves emotional regulation",
            "Group therapy with others who have bipolar disorder",
            "Functional Remediation - addresses cognitive impairments"
        ],
        "apps": [
            {"name": "MoodCare", "url": "https://www.moodcare.in", "description": "Indian app for mood tracking and bipolar management"},
            {"name": "InnerHour", "url": "https://www.theinnerhour.com", "description": "Indian app with specific programs for mood disorders"},
            {"name": "Wysa", "url": "https://www.wysa.io", "description": "Indian AI chatbot for emotional support"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian mental health platform with bipolar specialists"},
            {"name": "Juno Clinic", "url": "https://www.junoclinic.com", "description": "Indian teletherapy app with psychiatrists"}
        ],
        "helplines": [
            {"name": "Bipolar India", "number": "Email contact only", "email": "info@bipolarindia.com"},
            {"name": "NIMHANS Toll-Free Mental Health Helpline", "number": "080-46110007"},
            {"name": "Vandrevala Foundation", "number": "1860-2662-345 / 1800-2333-330", "hours": "24x7"}
        ]
    },
    "Personality disorder": {
        "description": "Personality disorders involve patterns of thinking, functioning, and behaving that significantly differ from cultural expectations and cause persistent distress or problems in relationships and daily functioning.",
        "self_help": [
            "Practice mindfulness meditation daily to increase awareness of thoughts and emotions",
            "Keep a detailed journal of thoughts, emotions, and behavior patterns",
            "Learn about your specific personality disorder through reputable sources",
            "Practice emotional regulation techniques like STOP skill (Stop, Take a step back, Observe, Proceed mindfully)",
            "Build healthy relationships with clear boundaries",
            "Use self-soothing techniques during emotional distress",
            "Practice opposite action to counteract unhelpful emotional urges",
            "Develop a crisis plan for intense emotional episodes",
            "Work on interpersonal effectiveness skills",
            "Try yoga and meditation practices that promote self-awareness"
        ],
        "therapies": [
            "Dialectical Behavior Therapy (DBT) - gold standard for borderline personality disorder",
            "Schema Therapy - addresses early maladaptive schemas",
            "Mentalization-Based Therapy (MBT) - improves understanding of mental states",
            "Cognitive Behavioral Therapy (CBT) - addresses distorted thinking patterns",
            "Transference-Focused Psychotherapy - examines relationship patterns",
            "Systems Training for Emotional Predictability and Problem Solving (STEPPS)",
            "Group therapy focusing on interpersonal skills"
        ],
        "apps": [
            {"name": "MindHelpr", "url": "https://mindhelpr.com", "description": "Indian app offering personality assessment and therapy"},
            {"name": "InnerHour", "url": "https://www.theinnerhour.com", "description": "Indian app with personality-specific programs"},
            {"name": "Wysa", "url": "https://www.wysa.io", "description": "Indian AI chatbot with DBT-informed techniques"},
            {"name": "YourDOST", "url": "https://yourdost.com", "description": "Indian platform connecting with personality disorder specialists"},
            {"name": "Juno Clinic", "url": "https://www.junoclinic.com", "description": "Indian teletherapy app with psychiatrists"}
        ],
        "helplines": [
            {"name": "NIMHANS Toll-Free Mental Health Helpline", "number": "080-46110007"},
            {"name": "Vandrevala Foundation", "number": "1860-2662-345 / 1800-2333-330", "hours": "24x7"},
            {"name": "iCall", "number": "022-25521111", "hours": "Mon-Sat, 8 AM to 10 PM"}
        ]
    }
}

# Helpline numbers by state
HELPLINES = {
    "Assam": ["Sarathi 104 (24x7)"],
    "Chandigarh": ["Asha Helpline: +91 172 2735436, +91 172 2735446 (Mon–Sat: 8am–7pm)"],
    "Chhattisgarh": ["Arogya Seva: Health Care and Health Counseling Center: 104 (24x7)"],
    "Delhi": [
        "Sumaitri: +91 011 23389090 (Mon–Fri: 2pm–10pm; Sat–Sun: 10am–10pm)",
        "Snehi: +91 011 65978181 (Daily: 2pm–6pm)",
        "Sanjeevani: 01124311918, 01124318883 (Mon–Fri); 26862222, 26864488, 40769002 (Mon–Sat) (10am–5:30pm)",
        "Fortis Stress Helpline: +91 8376804102 (24x7)"
    ],
    "Goa": ["COOJ Mental Health Foundation: +91 8322252525, +91 9822562522 (Weekdays: 3pm–7pm)"],
    "Gujarat": [
        "Saath: +91 79 26305544, +91 79 26300222 (Daily: 1pm–7pm)",
        "Jeevan Aastha Helpline: 1800 233 3330 (24x7)"
    ],
    "Jammu and Kashmir": [
        "Kashmir Lifeline: 1800 180 7020 (Sun–Thu: 10am–5pm)",
        "The Sara: +91-9697-606060 (Daily: 10am–5pm)"
    ],
    "Jharkhand": [
        "Chikitsa Salah: Health Information Helpline: 104 (24x7)",
        "Jeevan Suicide Prevention Helpline: +91 0657 6453841, +91 0657 6555555 (Daily: 10am–6pm)"
    ],
    "Karnataka": [
        "Parivarthan Counseling: +91 7676602602 (Mon–Fri: 4pm–10pm)",
        "SAHAI: +91 080 25497777, 9886444075 (Mon–Sat: 10am–8pm)",
        "Sa-Mudra Yuva: +91 9880396331 (24x7)",
        "Arogya Sahayavani: 104 (24x7)"
    ],
    "Kerala": [
        "Thanal Suicide Prevention Centre: +91 0495 2760000 (Daily: 10am–6pm)",
        "Maithri Kochi: +91 484 2540530 (Daily: 10am–7pm)",
        "Pratheeksha: +91 0484 2448830 (Daily: 10am–6pm)",
        "Prathyasa: +91 480 2820091 (Irinjalakuda)",
        "Sanjeevani: +91 0471 2533900 (Mon–Sat: 1pm–5pm)",
        "DISHA: 1056 (24x7)"
    ],
    "Madhya Pradesh": [
        "Spandan: +91 9630899002, +91 7389366696 (24x7)",
        "Sanjivani: 1253, +91 0761-2626622 (Jabalpur)",
        "Jeevan Aadhar - Adolescent Helpline: 1800-233-1250 (9am–5pm, except holidays)"
    ],
    "Maharashtra": [
        "Hitguj Help Number: +91 022 24131212 (Mumbai)",
        "Aasra: +91 9820466726 (24x7, Navi Mumbai)",
        "Nagpur Suicide Prevention Helpline: 8888817666",
        "Connecting NGO: 1800 843 4353 / 9922001122 (12pm–8pm, Pune)",
        "Vandrevala Foundation: 1860 266 2345, 1800 233 3330 (24x7)",
        "TISS iCALL: 022 25521111 (Mon–Sat: 8am–10pm)",
        "The Samaritans Mumbai: +91 84229 84528 / 84529 / 84530 (3pm–9pm)",
        "Maitra Helpline: +91 022 25385447 (Mon–Sat: 9am–9pm; Sun: 9am–1pm)",
        "Shushrusha Counseling: +91 9422627571, +91 8275038382 (24x7, Islampur)"
    ],
    "Odisha": ["Health Helpline: 104 (24x7)"],
    "Pondicherry": ["Maitreyi: +91 0413 2339999 (2pm–8pm)"],
    "Punjab": ["Medical Consultation – Health: 104 (24x7)"],
    "Rajasthan": [
        "Medical Advice & Helpline: 104 (24x7)",
        "Hope Helpline for Students: +91 0744 2333666, +91 0744 2414141 (24x7, Kota)"
    ],
    "Sikkim": ["Suicide Prevention Helpline: 221152, 18003453225 (24x7, Gangtok)"],
    "Tamil Nadu": [
        "Sneha India Foundation: +91 044-24640050 (24x7), +91 044-24640060 (8am–10pm)",
        "Medical Helpline: 104 (24x7)",
        "Jeevan Suicide Prevention Hotline: +91 044 2656 4444 (24x7)"
    ],
    "Telangana": [
        "Roshni Trust: +91 40 6620 2000, +91 40 6620 2001 (Mon–Sat: 11am–9pm)",
        "One Life: +91 7893078930 (24x7)",
        "Sevakendram Health Info: 104 (24x7)",
        "Darshika: +91 040 27755506, 040 27755505 (Secunderabad)",
        "Makro Foundation Helpdesk: +91 040 46004600 (Mon–Fri: 10am–7pm)"
    ],
    "West Bengal": [
        "Lifeline Foundation: +91 033 24637401, 24637432 (10am–6pm, Kolkata)",
        "NIBS Helpline: +91 98364 01234, +91 033 2286 5603 (Mon–Fri: 6pm–10pm)"
    ]
}

# Dictionary to store national mental health organizations for each condition
NATIONAL_ORGANIZATIONS = {
    "Anxiety": [
        "Anxiety and Depression Association of America (ADAA)",
        "National Alliance on Mental Illness (NAMI)",
        "Indian Association of Clinical Psychologists (IACP)"
    ],
    "Depression": [
        "Depression and Bipolar Support Alliance (DBSA)",
        "National Institute of Mental Health (NIMH)",
        "The Live Love Laugh Foundation"
    ],
    "Suicidal": [
        "American Foundation for Suicide Prevention",
        "International Association for Suicide Prevention",
        "Aasra Suicide Prevention"
    ],
    "Stress": [
        "The American Institute of Stress",
        "The Stress Management Society",
        "Mind (UK)"
    ],
    "Bipolar": [
        "Depression and Bipolar Support Alliance (DBSA)",
        "International Bipolar Foundation",
        "Bipolar India"
    ],
    "Personality disorder": [
        "National Education Alliance for Borderline Personality Disorder",
        "International Society for the Study of Personality Disorders",
        "Mind (UK)",
        "Treatment and Research Advancements Association for Personality Disorder"
    ],
    "Normal": [
        "Mental Health Foundation",
        "World Health Organization (WHO) Mental Health",
        "American Psychological Association"
    ]
}

def load_models():
    """Load all models and tokenizers in a thread-safe manner"""
    global models_loaded
    
    with loading_lock:
        if models_loaded:
            return
        
        try:
            logger.info(f"Loading models on device: {device}")
            for name, path in MODELS.items():
                logger.info(f"Loading model: {name} from {path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
                    loaded_models[name] = {"tokenizer": tokenizer, "model": model}
                    logger.info(f"Successfully loaded model: {name}")
                except Exception as e:
                    logger.error(f"Error loading model {name}: {str(e)}")
            
            if loaded_models:
                models_loaded = True
                logger.info(f"Successfully loaded {len(loaded_models)} models")
            else:
                logger.critical("No models were loaded successfully")
        except Exception as e:
            logger.error(f"Error in load_models: {str(e)}")
            raise

def ensure_models_loaded():
    """Ensure models are loaded before handling requests"""
    if not models_loaded:
        threading.Thread(target=load_models).start()
        return False
    return True

# Inject current datetime globally
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def sanitize_text(text):
    """Sanitize input text to prevent potential injection attacks or issues"""
    # Remove any HTML/script tags
    text = re.sub(r'<[^>]*>', '', text)
    return text.strip()

# Create a log of analyses for later review (anonymized)
def log_analysis(text, model_name, prediction, confidence):
    """Log analysis results to a file (anonymized)"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "text_length": len(text),
            "text_sample": text[:30] + "..." if len(text) > 30 else text  # Store only snippet for privacy
        }
        
        with open("analysis_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Error logging analysis: {str(e)}")

def predict(text: str, model_name: str):
    """Predict mental health condition from text using specified model"""
    try:
        if not ensure_models_loaded():
            return "Models are still loading. Please try again in a moment.", 0.0
            
        model_data = loaded_models.get(model_name)
        if not model_data:
            logger.error(f"Model not found: {model_name}")
            return "Model not found.", 0.0
            
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        
        # Ensure text is not empty and sanitize it
        text = sanitize_text(text)
        if not text:
            return "Empty text provided.", 0.0
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get prediction and confidence scores
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        class_id = torch.argmax(logits, dim=1).item()
        confidence = probabilities[class_id].item()
        
        prediction = model.config.id2label[class_id]
        
        # Log the analysis (anonymized)
        log_analysis(text, model_name, prediction, confidence)
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return f"Error: {str(e)}", 0.0

# Decorator for routes that require models to be loaded
def require_models(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not models_loaded:
            flash("Models are still loading. Please wait a moment and try again.", "warning")
            threading.Thread(target=load_models).start()
            return render_template("loading.html")
        return f(*args, **kwargs)
    return decorated_function
  

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for the application's homepage"""
    # Start loading models in the background if they're not loaded yet
    if not models_loaded:
        threading.Thread(target=load_models).start()
    
    text = request.form.get("text", "")
    selected_model = request.form.get("model") or next(iter(MODELS.keys()))
    
    # Initialize variables
    prediction = None
    confidence = None
    selected_state = None
    helpline_info = None
    resources_info = None
    
    if request.method == "POST" and text.strip():
        if not models_loaded:
            flash("Models are still loading. Please wait a moment and try again.", "warning")
            return render_template("index.html", 
                                   models=list(MODELS.keys()),
                                   selected_model=selected_model,
                                   states=sorted(HELPLINES.keys()),
                                   text=text,
                                   loading=True)
        
        # Sanitize input
        text = sanitize_text(text)
        
        prediction_result = predict(text, selected_model)
        if prediction_result and isinstance(prediction_result, tuple):
            prediction, confidence = prediction_result
            
            # Store prediction in session for history tracking
            if "history" not in session:
                session["history"] = []
                
            # Add to history, limiting to most recent 10 entries
            history_entry = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "model": selected_model
            }
            session["history"] = [history_entry] + session.get("history", [])[:9]
            session.modified = True
            
            # Get resources for the predicted condition
            resources_info = RESOURCES.get(prediction, {})
            
            # For Suicidal prediction, get state-specific helplines
            if prediction == "Suicidal":
                selected_state = request.form.get("state")
                helpline_info = HELPLINES.get(selected_state) if selected_state else None

    return render_template(
        "index.html",
        text=text,
        prediction=prediction,
        confidence=confidence * 100 if confidence else None,  # Convert to percentage
        states=sorted(HELPLINES.keys()),
        selected_state=selected_state,
        helpline_info=helpline_info,
        models=list(MODELS.keys()),
        selected_model=selected_model,
        resources=resources_info,
        national_orgs=NATIONAL_ORGANIZATIONS.get(prediction) if prediction else None,
        history=session.get("history", []),
        loading=not models_loaded
    )

@app.route("/api/analyze", methods=["POST"])
@require_models
def api_analyze():
    """API endpoint for analyzing text"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing text parameter"}), 400
            
        text = sanitize_text(data.get("text", ""))
        model_name = data.get("model", next(iter(MODELS.keys())))
        
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
            
        prediction, confidence = predict(text, model_name)
        
        return jsonify({
            "prediction": prediction,
            "confidence": confidence * 100,  # Convert to percentage
            "resources": RESOURCES.get(prediction, {}),
            "national_orgs": NATIONAL_ORGANIZATIONS.get(prediction, [])
        })
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/resources/<condition>")
def resources(condition):
    """Route to show detailed resources for a specific condition"""
    condition = condition.capitalize()
    if condition not in RESOURCES:
        flash("Invalid condition specified", "error")
        return redirect(url_for("index"))
        
    resources_info = RESOURCES.get(condition, {})
    national_orgs = NATIONAL_ORGANIZATIONS.get(condition, [])
    
    # For Suicidal condition, include state helplines
    helplines = None
    if condition == "Suicidal":
        helplines = HELPLINES
        
    return render_template(
        "resources.html",
        condition=condition,
        resources=resources_info,
        national_orgs=national_orgs,
        helplines=helplines
    )

@app.route("/history")
def history():
    """Route to show analysis history"""
    return render_template(
        "history.html",
        history=session.get("history", [])
    )

@app.route("/about")
def about():
    """Route to show information about the application"""
    return render_template("about.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    """Route to collect user feedback on predictions"""
    if request.method == "POST":
        feedback_text = sanitize_text(request.form.get("feedback", ""))
        prediction = request.form.get("prediction")
        accurate = request.form.get("accurate")
        
        # Log the feedback
        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "accurate": accurate == "yes",
                "feedback": feedback_text
            }
            
            with open("feedback_log.json", "a") as f:
                f.write(json.dumps(feedback_entry) + "\n")
                
            flash("Thank you for your feedback!", "success")
            return redirect(url_for("index"))
            
        except Exception as e:
            logger.error(f"Error logging feedback: {str(e)}")
            flash("Error saving feedback. Please try again.", "error")
    
    return render_template("feedback.html")

@app.route("/clear_history")
def clear_history():
    """Clear the user's history from the session"""
    if "history" in session:
        session.pop("history")
        flash("History cleared successfully", "success")
    return redirect(url_for("history"))

@app.route("/api/models/status")
def models_status():
    """Check if models are loaded"""
    return jsonify({"loaded": models_loaded, "count": len(loaded_models)})

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template("500.html"), 500


@app.route("/health")
def health():
    return render_template(
        "health.html",
        status="ok",
        models_loaded=models_loaded,
        model_count=len(loaded_models)
    ), 200    


# Setup app initialization code
def init_app():
    """Initialize the application"""
    # Create necessary directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    
    # Ensure log files exist
    for log_file in ["analysis_log.json", "feedback_log.json"]:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("")
    
    # Start loading models in background
    threading.Thread(target=load_models).start()

if __name__ == "__main__":
    try:
        init_app()
        logger.info("Starting Flask application")
        app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)

    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")




        