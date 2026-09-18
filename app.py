import os
import hashlib
import json
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if present
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import gradio as gr

# ============= CRISIS DETECTION SYSTEM =============

CRISIS_KEYWORDS = [
    'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
    'better off dead', 'no reason to live', 'self harm', 'hurt myself',
    'end it all', 'can\'t go on', 'not worth living', 'take my life',
    'kill me', 'die', 'death wish', 'overdose', 'jump off', 'hang myself',
    'cutting myself', 'worthless', 'burden to everyone', 'everyone would be better',
    'goodbye forever', 'last goodbye', 'final message', 'can\'t take it anymore'
]

def detect_crisis(message):
    """Detect crisis keywords in user message."""
    message_lower = message.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in message_lower:
            return True
    return False

def send_crisis_alert(username, message, trusted_email):
    """Send email alert to trusted contact."""
    users = load_users()
    user_info = users.get(username, {})

    print(f"\n{'='*70}")
    print("🚨 CRISIS ALERT TRIGGERED 🚨")
    print(f"{'='*70}")
    print(f"User: {username}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Message: {message[:100]}...")
    print(f"📧 Sending email to: {trusted_email}")
    print(f"{'='*70}\n")

    log_crisis_alert(username, message, trusted_email)

    try:
        email_sent = send_email_alert(username, message, trusted_email, user_info)
        if email_sent:
            print("✅ Crisis alert email sent successfully!")
            return True
        else:
            print("❌ Failed to send email alert")
            return False
    except Exception as e:
        print(f"❌ Error sending alert: {e}")
        return False

def log_crisis_alert(username, message, trusted_email):
    """Log all crisis alerts to JSON file."""
    log_file = "crisis_alerts.json"

    alert_data = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "message": message,
        "trusted_email": trusted_email,
        "status": "logged"
    }

    alerts = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            alerts = []

    alerts.append(alert_data)

    with open(log_file, 'w') as f:
        json.dump(alerts, f, indent=2)

    print(f"✅ Crisis alert logged to {log_file}")

def send_email_alert(username, message, trusted_email, user_info):
    """Send email alert to trusted contact."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # EMAIL CONFIGURATION (loaded from environment variables — see .env.example)
        sender_email = os.environ.get("GMAIL_SENDER_EMAIL")
        sender_password = os.environ.get("GMAIL_APP_PASSWORD")
        if not sender_email or not sender_password:
            print("⚠️ Email not configured. Set GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD.")
            return False

        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = trusted_email
        msg['Subject'] = '🚨 URGENT: Crisis Alert - Smart Emotional ChatBot.'

        # Plain text version
        text_body = f"""
🚨 CRISIS ALERT - Smart Emotional ChatBot 🚨

URGENT: Someone needs your immediate support.

User Details:
- Username: {username}
- Phone: {user_info.get('phone', 'N/A')}
- Country: {user_info.get('country', 'N/A')}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Concerning Message Detected:
\"{message[:200]}...\"\n
⚠️ IMMEDIATE ACTION REQUIRED ⚠️

This person may be in crisis. Please:
1. Call them IMMEDIATELY at: {user_info.get('phone', 'N/A')}
2. If you cannot reach them, call emergency services
3. Stay with them or ensure someone is with them

Emergency Resources:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDIA:
• KIRAN Mental Health: 1800-599-0019
• Vandrevala Foundation: 1860-2662-345
• iCall: 9152987821
• Emergency: 112

UNITED STATES:
• Suicide Prevention: 988
• Crisis Text Line: Text HOME to 741741
• Emergency: 911

INTERNATIONAL:
• https://findahelpline.com
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is an automated alert from Smart Emotional ChatBot.

DO NOT IGNORE THIS MESSAGE - A life may depend on your response.
"""

        # HTML version
        html_body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 650px; margin: 0 auto; background-color: #f5f5f5;">
            <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; padding: 30px 20px; text-align: center; border-radius: 8px 8px 0 0;">
                <h1 style="margin: 0; font-size: 28px; font-weight: 700;">🚨 CRISIS ALERT</h1>
                <p style="font-size: 16px; margin: 10px 0 0 0; opacity: 0.95;">Smart Emotional ChatBot System</p>
            </div>

            <div style="padding: 25px; background: #fff3cd; border-left: 5px solid #ffc107;">
                <h2 style="color: #856404; margin-top: 0; font-size: 20px;">⚠️ URGENT: Immediate Action Required</h2>
                <p style="margin: 0; font-size: 15px; line-height: 1.6;"><strong>Someone you care about may be in crisis and needs your support right now.</strong></p>
            </div>

            <div style="padding: 30px; background: white;">
                <h3 style="color: #333; font-size: 18px; margin-top: 30px; border-bottom: 2px solid #667eea; padding-bottom: 10px;">User Information</h3>
                <table style="width: 100%; line-height: 2; font-size: 14px;">
                    <tr><td style="color: #666; width: 30%;"><strong>Username:</strong></td><td>{username}</td></tr>
                    <tr><td style="color: #666;"><strong>Phone:</strong></td><td>{user_info.get('phone', 'N/A')}</td></tr>
                    <tr><td style="color: #666;"><strong>Country:</strong></td><td>{user_info.get('country', 'N/A')}</td></tr>
                    <tr><td style="color: #666;"><strong>Time:</strong></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                </table>

                <h3 style="color: #dc3545; font-size: 18px; margin-top: 30px; border-bottom: 2px solid #dc3545; padding-bottom: 10px;">Concerning Message Detected</h3>
                <div style="background: #f8f9fa; padding: 20px; border-left: 4px solid #dc3545; font-style: italic; border-radius: 4px; margin: 15px 0;">
                    <p style="margin: 0; color: #333; line-height: 1.6;">\"{message[:200]}...\"</p>
                </div>

                <h3 style="color: #dc3545; margin-top: 35px; font-size: 18px;">What You Should Do NOW</h3>
                <ol style="line-height: 2; color: #333; font-size: 14px; padding-left: 20px;">
                    <li><strong>Call them immediately</strong> at: <a href="tel:{user_info.get('phone', '')}" style="color: #667eea; text-decoration: none; font-weight: 600;">{user_info.get('phone', 'N/A')}</a></li>
                    <li>If you cannot reach them, <strong>call emergency services</strong></li>
                    <li>Stay with them or ensure someone is with them</li>
                    <li>Listen without judgment and take their feelings seriously</li>
                    <li>Help them access professional support</li>
                </ol>

                <div style="background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding: 25px; border-radius: 8px; margin: 30px 0;">
                    <h3 style="color: #0c5460; margin-top: 0; font-size: 18px;">📞 Emergency Resources</h3>

                    <div style="margin: 20px 0;">
                        <p style="margin: 5px 0; font-weight: 700; color: #0c5460; font-size: 15px;">🇮🇳 INDIA</p>
                        <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                            <li>KIRAN Mental Health: <strong style="color: #0c5460;">1800-599-0019</strong></li>
                            <li>Vandrevala Foundation: <strong style="color: #0c5460;">1860-2662-345</strong></li>
                            <li>iCall (Mumbai): <strong style="color: #0c5460;">9152987821</strong></li>
                            <li>Emergency Services: <strong style="color: #0c5460;">112</strong></li>
                        </ul>
                    </div>

                    <div style="margin: 20px 0;">
                        <p style="margin: 5px 0; font-weight: 700; color: #0c5460; font-size: 15px;">🇺🇸 UNITED STATES</p>
                        <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                            <li>Suicide Prevention Lifeline: <strong style="color: #0c5460;">988</strong></li>
                            <li>Crisis Text Line: Text <strong style="color: #0c5460;">HOME</strong> to <strong style="color: #0c5460;">741741</strong></li>
                            <li>Emergency: <strong style="color: #0c5460;">911</strong></li>
                        </ul>
                    </div>

                    <p style="margin: 15px 0 0 0;"><strong>🌍 INTERNATIONAL:</strong> <a href="https://findahelpline.com" style="color: #0c5460; font-weight: 600;">findahelpline.com</a></p>
                </div>

                <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 20px; border-radius: 8px; border-left: 5px solid #dc3545;">
                    <p style="margin: 0; color: #721c24; font-weight: 700; font-size: 15px;">⚠️ DO NOT IGNORE THIS MESSAGE</p>
                    <p style="margin: 10px 0 0 0; color: #721c24;">A life may depend on your response. Please act immediately and contact them now.</p>
                </div>
            </div>

            <div style="background: #e9ecef; padding: 20px; font-size: 12px; color: #6c757d; text-align: center; border-radius: 0 0 8px 8px;">
                <p style="margin: 5px 0;">This is an automated crisis alert from <strong>Smart Emotional ChatBot</strong></p>
                <p style="margin: 5px 0;">Alert logged at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p style="margin: 5px 0;">For technical support, please contact your system administrator</p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        print(f"📧 Sending email to: {trusted_email}")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("✅ EMAIL SENT!")
        return True

    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

# ============= USER AUTHENTICATION =============

USER_DB_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_users(users):
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    print(f"✅ Users saved to {USER_DB_FILE}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def reset_user_password(username, new_password):
    """Admin function to reset a user's password"""
    users = load_users()
    if username in users:
        users[username]["password"] = hash_password(new_password)
        save_users(users)
        print(f"✅ Password reset for {username}")
        return True
    return False

def view_all_users():
    """Debug function to view all registered users"""
    users = load_users()
    print("\n" + "="*70)
    print("REGISTERED USERS:")
    print("="*70)
    for username, data in users.items():
        print(f"Username: {username}")
        print(f"  Email: {data.get('email', 'N/A')}")
        print(f"  Country: {data.get('country', 'N/A')}")
        print(f"  Phone: {data.get('phone', 'N/A')}")
        print(f"  Trusted Email: {data.get('trusted_email', 'N/A')}")
        print(f"  Created: {data.get('created_at', 'N/A')}")
        print("-"*70)
    print(f"Total users: {len(users)}\n")

def validate_phone(phone):
    if not phone:
        return False
    phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
    return bool(re.match(r'^\+?[0-9]{10,15}$', phone_clean))

def validate_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def signup_user(username, password, email, country, phone, trusted_email):
    if not all([username, password, email, country, phone, trusted_email]):
        return "❌ All fields required!", None

    users = load_users()

    if username in users:
        return "❌ Username exists!", None

    if len(password) < 6:
        return "❌ Password too short!", None

    if not validate_email(email):
        return "❌ Invalid email!", None

    if not validate_email(trusted_email):
        return "❌ Invalid trusted email!", None

    if email == trusted_email:
        return "❌ Emails must differ!", None

    if not validate_phone(phone):
        return "❌ Invalid phone format!", None

    users[username] = {
        "password": hash_password(password),
        "email": email,
        "country": country,
        "phone": phone,
        "trusted_email": trusted_email,
        "created_at": datetime.now().isoformat()
    }

    save_users(users)
    print(f"✅ User registered: {username}")
    return "✅ Account created! Please login.", None

def login_user(username, password):
    if not username or not password:
        return "❌ Username and password required!", None

    users = load_users()

    if username not in users:
        return "❌ Invalid credentials!", None

    stored_password = users[username]["password"]
    input_password_hash = hash_password(password)

    # Debug logging
    print(f"Login attempt for: {username}")
    print(f"Stored hash: {stored_password}")
    print(f"Input hash: {input_password_hash}")
    print(f"Match: {stored_password == input_password_hash}")

    if stored_password != input_password_hash:
        return "❌ Invalid credentials!", None

    print(f"✅ User logged in: {username}")
    return f"✅ Welcome, {username}!", username

# ============= CHATBOT FUNCTIONS =============

def initialize_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

def create_vector_db():
    loader = DirectoryLoader("./data", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are the Smart Emotional ChatBot. Respond truthfully.

    Context: {context}
    Question: {question}

    Helpful Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

def chatbot_response(message, history, username):
    try:
        if not message.strip():
            return "Please provide input."

        if detect_crisis(message):
            print(f"⚠️ Crisis detected: {username}")
            users = load_users()
            if username and username in users:
                trusted_email = users[username].get("trusted_email", "")
                if trusted_email:
                    send_crisis_alert(username, message, trusted_email)

                return """🆘 I notice concerning thoughts. Your safety matters most.\n\n**🚨 Immediate Help:**\n• US: 988 | Text HOME to 741741\n• India: 1800-599-0019\n• International: findahelpline.com\n
**🔔 Your trusted contact notified**

**💙 You're not alone**
Would you like to talk about what's troubling you?"""

        response = qa_chain.run(message)
        return response

    except Exception as e:
        print(f"Error: {e}")
        return "I apologize for the error. Please try again."

# ============= INITIALIZE =============

print("Initializing...")

llm = None
try:
    llm = initialize_llm()
    print("✅ LLM ready")
except Exception as e:
    print(f"Error: {e}")

db_path = "./chroma_db"

if not os.path.exists(db_path):
    try:
        vector_db = create_vector_db()
    except:
        vector_db = None
else:
    try:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        print("✅ DB loaded")
    except:
        vector_db = None

qa_chain = None
if llm and vector_db:
    try:
        qa_chain = setup_qa_chain(vector_db, llm)
        print("✅ QA chain ready")
    except Exception as e:
        print(f"Error: {e}")

# ============= UI WITH ENHANCED STYLING =============

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* Animated gradient background for auth page */
.auth-container {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
    padding: 60px 20px;
    position: relative;
    overflow: hidden;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating shapes in background */
.auth-container::before {
    content: '';
    position: absolute;
    width: 300px;
    height: 300px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    top: -100px;
    left: -100px;
    animation: float 20s infinite;
}

.auth-container::after {
    content: '';
    position: absolute;
    width: 400px;
    height: 400px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 50%;
    bottom: -150px;
    right: -150px;
    animation: float 25s infinite reverse;
}

@keyframes float {
    0%, 100% { transform: translateY(0px) translateX(0px); }
    50% { transform: translateY(50px) translateX(30px); }
}

/* Glass morphism effect for auth box */
.auth-box {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 48px;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(255, 255, 255, 0.5);
    max-width: 520px;
    margin: 0 auto;
    position: relative;
    z-index: 10;
    animation: slideIn 0.6s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Visible text styles */
.auth-box h1, .auth-box h2, .auth-box h3, .auth-box h4, .auth-box p {
    color: #1a1a2e !important;
}

.auth-box label {
    color: #2d3748 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

/* Gradient text with brain icon */
.gradient-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 32px !important;
    margin-bottom: 4px !important;
    text-align: center;
    display: block;
}

.subtitle {
    color: #4a5568 !important;
    font-size: 16px !important;
    text-align: center;
    margin-bottom: 32px !important;
    font-weight: 500 !important;
}

/* Chatbot container with subtle pattern */
.chatbot-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    background-image:
        linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    min-height: 100vh;
    padding: 30px 20px;
}

.chat-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.chat-header h1 {
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 28px !important;
    margin: 0 !important;
}

.chat-header p {
    color: #4a5568 !important;
    font-size: 14px !important;
    margin: 4px 0 0 0 !important;
}

/* Enhanced buttons */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Info cards with icons and better visibility */
.info-card {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #2196f3;
    padding: 18px;
    border-radius: 12px;
    margin: 20px 0;
    color: #0d47a1 !important;
    box-shadow: 0 2px 8px rgba(33, 150, 243, 0.2);
}

.info-card strong {
    color: #01579b !important;
    font-size: 15px !important;
}

.warning-card {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 5px solid #ff9800;
    padding: 18px;
    border-radius: 12px;
    margin: 20px 0;
    color: #e65100 !important;
    box-shadow: 0 2px 8px rgba(255, 152, 0, 0.2);
}

.warning-card strong {
    color: #bf360c !important;
    font-size: 15px !important;
}

.crisis-card {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 5px solid #f44336;
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    color: #b71c1c !important;
    box-shadow: 0 2px 8px rgba(244, 67, 54, 0.2);
}

.crisis-card strong {
    color: #b71c1c !important;
    font-size: 16px !important;
    display: block;
    margin-bottom: 8px;
}

.crisis-card a {
    color: #1565c0 !important;
    text-decoration: underline !important;
    font-weight: 600;
}

.crisis-card small {
    color: #c62828 !important;
    font-size: 13px !important;
}

/* Tab styling */
.tab-nav button {
    color: #4a5568 !important;
    font-weight: 500 !important;
    font-size: 16px !important;
    padding: 12px 24px !important;
}

.tab-nav button[aria-selected="true"] {
    color: #667eea !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #667eea !important;
}

/* Input fields */
input, textarea, select {
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Info text below inputs */
.info, .hint {
    color: #718096 !important;
    font-size: 13px !important;
    margin-top: 4px !important;
}

/* Message status */
.textbox {
    color: #1a1a2e !important;
    font-weight: 500 !important;
}

/* Chat interface */
.message {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    margin: 8px 0 !important;
}

.message.user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.message.bot {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    color: #1a1a2e !important;
}

/* Fix chatbot message visibility */
.chatbot .message-wrap {
    background: white !important;
}

.chatbot .message-wrap .message {
    color: #1a1a2e !important;
}

.chatbot .bot {
    background: #f7fafc !important;
    border: 1px solid #e2e8f0 !important;
}

.chatbot .bot p, .chatbot .bot span, .chatbot .bot div {
    color: #1a1a2e !important;
}

.chatbot .user {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.chatbot .user p, .chatbot .user span, .chatbot .user div {
    color: white !important;
}

/* Ensure all text in chatbot is visible */
.chatbot {
    background: white !important;
}

.chatbot * {
    color: #1a1a2e !important;
}

.chatbot .user * {
    color: white !important;
}

/* Specific Gradio chatbot styling */
.prose p, .prose li, .prose span, .prose strong, .prose em {
    color: #1a1a2e !important;
}

.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #1a1a2e !important;
}

/* Bot message container */
[data-testid="bot"] {
    background: #f7fafc !important;
    color: #1a1a2e !important;
}

[data-testid="bot"] * {
    color: #1a1a2e !important;
}

/* User message container */
[data-testid="user"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

[data-testid="user"] * {
    color: white !important;
}

/* Logout button */
button[size="sm"] {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 2px solid #667eea !important;
    color: #667eea !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    transition: all 0.3s ease !important;
}

button[size="sm"]:hover {
    background: #667eea !important;
    color: white !important;
}

/* Examples */
.examples {
    margin-top: 16px !important;
}

.examples button {
    background: white !important;
    border: 2px solid #538edb !important;
    color: #4a5568 !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
    transition: all 0.3s ease !important;
}

.examples button:hover {
    border-color: #667eea !important;
    color: #667eea !important;
    background: #f7fafc !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="indigo"), title="Smart Emotional ChatBot") as app:
    current_user = gr.State(None)

    with gr.Column(elem_classes="auth-container") as auth_page:
        with gr.Column(elem_classes="auth-box"):
            gr.HTML("<h1 class='gradient-text'>🧠 Smart Emotional ChatBot</h1>")
            gr.HTML("<p class='subtitle'>Professional AI Mental Wellness Support</p>")

            with gr.Tabs():
                with gr.Tab("🔐 Login"):
                    gr.Markdown("<h3 style='color: #1a1a2e; text-align: center; margin-bottom: 24px;'>Welcome Back</h3>")
                    login_username = gr.Textbox(label="Username", placeholder="Enter username")
                    login_password = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                    login_btn = gr.Button("Login", elem_classes="primary-btn", variant="primary")
                    login_msg = gr.Textbox(label="", interactive=False, show_label=False)

                with gr.Tab("✨ Create Account"):
                    gr.Markdown("<h3 style='color: #1a1a2e; text-align: center; margin-bottom: 16px;'>Join Smart Emotional ChatBot</h3>")
                    gr.HTML('<div class="info-card"><strong>🛡️ Privacy:</strong> Your data is secure and confidential.</div>')

                    signup_username = gr.Textbox(label="Username", placeholder="Choose username")
                    signup_email = gr.Textbox(label="Your Email", placeholder="your.email@example.com")
                    signup_password = gr.Textbox(label="Password", placeholder="Minimum 6 characters", type="password")
                    signup_country = gr.Dropdown(label="Country", choices=["United States", "United Kingdom", "Canada", "Australia", "India", "Germany", "France", "Japan", "Other"], value="United States")
                    signup_phone = gr.Textbox(label="Phone", placeholder="+1234567890")

                    gr.Markdown("<h3 style='color: #1a1a2e; margin-top: 24px;'> Trusted person Contact</h3>")


                    signup_trusted_email = gr.Textbox(label="Trusted Contact Email", placeholder="Trusted person@example.com", info="Receives crisis alerts")

                    signup_btn = gr.Button("Create Account", elem_classes="primary-btn", variant="primary")
                    signup_msg = gr.Textbox(label="", interactive=False, show_label=False)

    with gr.Column(visible=False, elem_classes="chatbot-container") as chatbot_page:
        with gr.Column(elem_classes="chat-header"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.HTML("<h1>🧠 Smart Emotional Chatbot</h1><p>Confidential Mental Wellness Support</p>")
                with gr.Column(scale=1):
                    logout_btn = gr.Button("Logout", size="sm")



        chatbot_interface = gr.Chatbot(
            label="Chat",
            height=500,
            bubble_full_width=False,
            show_label=False
        )
        msg_input = gr.Textbox(label="Message", placeholder="Type your message here...", lines=2)

        with gr.Row():
            send_btn = gr.Button("📤 Send", variant="primary", elem_classes="primary-btn", scale=2)
            clear_btn = gr.Button("🗑️ Clear", scale=1)

        gr.Examples(
            ["How can I manage stress?", "What are relaxation techniques?", "I'm feeling overwhelmed", "Ways to improve sleep?"],
            inputs=msg_input,
            label="💡 Example Questions"
        )

    # ============= EVENT HANDLERS =============

    def handle_login(username, password):
        msg, user = login_user(username, password)
        if user:
            return msg, user, gr.update(visible=False), gr.update(visible=True), []
        return msg, None, gr.update(visible=True), gr.update(visible=False), []

    def handle_signup(username, password, email, country, phone, trusted_email):
        msg, _ = signup_user(username, password, email, country, phone, trusted_email)
        return msg, None

    def handle_logout():
        return None, gr.update(visible=True), gr.update(visible=False), []

    def respond(message, chat_history, username):
        if not message.strip():
            return chat_history, ""
        bot_message = chatbot_response(message, chat_history, username)
        chat_history.append((message, bot_message))
        return chat_history, ""

    # Connect event handlers
    login_btn.click(
        handle_login,
        [login_username, login_password],
        [login_msg, current_user, auth_page, chatbot_page, chatbot_interface]
    )

    signup_btn.click(
        handle_signup,
        [signup_username, signup_password, signup_email, signup_country, signup_phone, signup_trusted_email],
        [signup_msg, current_user]
    )

    logout_btn.click(
        handle_logout,
        outputs=[current_user, auth_page, chatbot_page, chatbot_interface]
    )

    send_btn.click(
        respond,
        [msg_input, chatbot_interface, current_user],
        [chatbot_interface, msg_input]
    )

    msg_input.submit(
        respond,
        [msg_input, chatbot_interface, current_user],
        [chatbot_interface, msg_input]
    )

    clear_btn.click(lambda: [], outputs=chatbot_interface)

# ============= LAUNCH APPLICATION =============

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 SMART EMOTIONAL CHATBOT - Starting Application")
    print("="*70)
    print("✅ System initialized")
    print("✅ Crisis detection active")
    print("✅ User authentication ready")

    # Show registered users for debugging
    view_all_users()

    print("="*70 + "\n")

    app.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

    print("\n✅ Application launched successfully!")
    print("🌐 Access the app through the provided URL")
    print("="*70)
