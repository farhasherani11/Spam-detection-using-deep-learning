import pandas as pd

df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')
import pandas as pd

df = pd.read_csv("D:/Spam-detection-using-deep-learning/data/raw/spam.csv", encoding='latin-1')

# Rename columns
df = df.rename(columns={
    'Category': 'v1',
    'Message': 'v2'
})


df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

spam_categories = [
    "promotion",     # ads, offers, discounts
    "financial",     # bank, loan, money
    "phishing",      # fake links, login traps
    "lottery",       # win prize, lucky draw
    "job_spam",      # job offers, work from home
    "otp_fraud",     # OTP/password scams
    "adult",         # inappropriate content
    "general_spam",  # spam but no clear type
    "ham"            # normal message
]
def classify_spam_type(text, label):
    text = text.lower()
    
    if label == "ham":
        return "ham"
    
    if "win" in text or "prize" in text or "lottery" in text:
        return "lottery"
    elif "loan" in text or "bank" in text or "money" in text:
        return "financial"
    elif "click" in text or "link" in text or "verify" in text:
        return "phishing"
    elif "job" in text or "earn" in text:
        return "job_spam"
    elif "otp" in text or "password" in text:
        return "otp_fraud"
    elif "free" in text or "offer" in text or "buy" in text:
        return "promotion"
    elif "sex" in text or "adult" in text:
        return "adult"
    else:
        return "general_spam"


df['spam_type'] = df.apply(lambda x: classify_spam_type(x['message'], x['label']), axis=1)


print(df.head(15))