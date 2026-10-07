"""
scripts/augment_dataset.py
==========================
Augments the UCI SMS Spam Collection with:
1. Synthetic modern HAM examples (OTPs, bank alerts, order confirmations, etc.)
2. Synthetic modern SPAM examples (phishing, prize scams)
3. SpamAssassin 2003 corpus (downloaded automatically)

Run:
    .venv\Scripts\python scripts/augment_dataset.py

Output:
    data/raw/spam_augmented.csv  -- merged, deduplicated, shuffled dataset
"""

import csv
import random
import tarfile
import urllib.request
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  TEMPLATE BANK
# ---------------------------------------------------------------------------

HAM_TEMPLATES = [
    # Google / Account services
    "Your Google Account {email} was recently signed in on a new device. If this was you, no action needed.",
    "Google: A sign-in from a new device was detected. Review activity at myaccount.google.com",
    "Reminder: Google Terms of Service have been updated. Please review them at policies.google.com",
    "Your Google storage is {pct}% full. Manage your storage at one.google.com",
    "Google Play: Your subscription to {service} renews on {date}.",
    "YouTube: {channel} posted a new video you might like.",
    "Security alert: Your Google password was changed on {date}. If you did not do this, visit g.co/recover",
    "Google Photos: Your memories from {n} years ago are ready to view.",

    # Microsoft / Office 365
    "Microsoft account: Security info was added to your account on {date}.",
    "Your Microsoft 365 subscription will renew on {date} for Rs. {amount}.",
    "Microsoft Teams: {name} shared a file with you in {channel}.",
    "OneDrive: Your storage is {pct}% full. Free up space or upgrade.",
    "Microsoft: You have {n} new notifications in your Teams channel.",
    "Outlook calendar reminder: Meeting '{title}' starts in 15 minutes.",

    # Apple
    "Apple ID: Your Apple ID was used to sign in to iCloud on a new device.",
    "Your Apple ID password was changed on {date}. If you did not make this change, go to iforgot.apple.com",
    "App Store: Your subscription to {service} will renew on {date}.",

    # Amazon / E-commerce
    "Your Amazon order #{order_id} has been shipped and will arrive by {date}.",
    "Amazon: Your package is out for delivery today. Track at amazon.in",
    "Your Amazon order has been delivered. How was your experience?",
    "Amazon: Payment of Rs.{amount} for order #{order_id} was successful.",
    "Flipkart: Your order #{order_id} for {product} has been shipped via {courier}.",
    "Myntra: Your order is on its way! Estimated delivery: {date}.",
    "Swiggy: Your order from {restaurant} is being prepared. Delivery in {min} minutes.",
    "Zomato: {restaurant} has accepted your order. Estimated delivery in {min} minutes.",
    "Meesho: Your order #{order_id} has been dispatched. Expected by {date}.",

    # Bank / Financial alerts
    "Your account XX{last4} has been debited Rs.{amount} on {date}. Available balance: Rs.{balance}.",
    "HDFC Bank: Rs.{amount} debited from A/c XX{last4} on {date} at {merchant}. UPI Ref:{ref}",
    "SBI: Rs.{amount} credited to your account XX{last4} on {date}.",
    "Kotak: ATM withdrawal of Rs.{amount} from your account XX{last4} on {date}.",
    "ICICI Bank: Your credit card XX{last4} bill of Rs.{amount} is due on {date}.",
    "Axis Bank: Your FD of Rs.{amount} has matured. Login to netbanking to renew.",
    "PayPal: You sent Rs.{amount} to {name}. Transaction ID: {ref}",
    "Paytm: Rs.{amount} paid successfully to {merchant}. UPI Ref: {ref}",
    "Google Pay: Rs.{amount} sent to {name} successfully via UPI.",
    "PhonePe: Rs.{amount} debited from your bank account for payment to {merchant}.",

    # OTP / Verification
    "{n} is your OTP for login to {service}. Valid for 10 minutes. Do not share with anyone.",
    "Your verification code for {service} is {n}. It expires in 5 minutes.",
    "{n} is your {service} OTP. Valid for 10 mins. Please do not share with anyone.",
    "OTP for your transaction of Rs.{amount} at {merchant} is {n}. DO NOT SHARE this OTP.",
    "{n} is the OTP for your account login. Use within 30 seconds.",
    "Your OTP is {n}. It is valid for 10 minutes. Please do not share this with anyone.",
    "Use OTP {n} to verify your mobile number on {service}. OTP valid for 10 minutes.",

    # Professional / Work emails
    "Hi {name}, please find attached the minutes of today's meeting. Let me know if you have edits.",
    "Hi team, the standup has been moved to {time} on {date}. Please update your calendars.",
    "Your pull request #{n} has been reviewed and approved. Merge when ready.",
    "Reminder: Your interview with {company} is scheduled for {date} at {time} via Google Meet.",
    "The Q{n} board report has been shared with you on Google Drive. Please review before {date}.",
    "Hi {name}, following up on the proposal I sent. Any questions or feedback?",
    "Your invoice #{n} for Rs.{amount} has been paid. Receipt attached.",
    "Action required: Please complete your performance review by {date}.",

    # Social media notifications
    "Reddit: {user} replied to your comment in r/{subreddit}.",
    "Reddit: Your post in r/{subreddit} reached {n} upvotes!",
    "LinkedIn: {name} accepted your connection request.",
    "LinkedIn: {company} is hiring {role} in {city}. {n} of your connections work there.",
    "Twitter: {user} retweeted your tweet.",
    "Instagram: {user} liked your recent photo.",
    "Instagram: {user} started following you.",

    # Education / Job platforms
    "Coursera: Your certificate for '{course}' is ready. Download from your dashboard.",
    "Internshala: New internship in {role} at {company} in {city}. Stipend: Rs.{amount}/month.",
    "NPTEL: Your assignment submission for Week {n} has been received successfully.",
    "GeeksforGeeks: You solved {n} problems this week. Keep it up!",
    "upGrad: Your next live session for '{course}' is on {date} at {time}.",

    # Travel / Utilities
    "IRCTC: Booking confirmed. PNR: {ref}. Train {train} on {date} from {city} to {city2}.",
    "IndiGo: Your booking {ref} is confirmed. Flight {flight} on {date}. Web check-in opens 48 hrs before.",
    "MakeMyTrip: Your hotel in {city} is confirmed. Check-in: {date}. Booking ID: {ref}",
    "Ola: Your ride has ended. Total: Rs.{amount}. Rate your ride.",
    "Uber: Your trip ended. Charged Rs.{amount} to your card XX{last4}.",

    # Government / Regulatory
    "Income Tax: Your ITR for AY {year} has been processed. Refund of Rs.{amount} initiated.",
    "Aadhaar: Your Aadhaar data was accessed by {service} on {date} for authentication.",
    "DigiLocker: Your document {doc} has been issued by {authority} and added to your account.",
    "EPFO: Rs.{amount} has been credited to your EPF account for the month of {month}.",

    # GitHub / Developer tools
    "GitHub: {user} opened a new issue in {repo}: '{title}'",
    "GitHub Actions: Workflow '{workflow}' in {repo} completed successfully.",
    "Jira: {user} assigned PROJ-{n} to you. Summary: {title}",
    "Slack: You have {n} unread messages in #{channel}",
    "Bitbucket: {user} approved your pull request in {repo}.",

    # Healthcare
    "Your appointment with Dr. {name} is confirmed for {date} at {time}. Reply CANCEL to cancel.",
    "Prescription reminder: Time to refill your {service} prescription. Visit the pharmacy.",
    "Your lab reports for {date} are now available on the hospital portal.",

    # Utilities / Bills
    "Your electricity bill for {month} is Rs.{amount}. Due date: {date}. Pay at bescom.karnataka.gov.in",
    "Airtel: Your mobile bill for {month} is Rs.{amount}. Due by {date}. Pay at airtel.in",
    "Jio: Your recharge of Rs.{amount} is successful. Validity: {n} days.",
    "BSNL: Your broadband bill for {month} is Rs.{amount}. Pay before {date} to avoid disconnection.",
]

SPAM_TEMPLATES = [
    # Prize / lottery
    "Congratulations! You have been selected to receive a FREE {prize}. Click {url} to claim NOW!",
    "You are a LUCKY WINNER! Rs.{amount} cash prize is waiting. Confirm at {url} immediately.",
    "WINNER ANNOUNCEMENT: You have won a {prize} worth Rs.{amount}! Claim within 24 hours at {url}",
    "You have been selected as winner of our lucky draw. Prize: {prize}. Call {phone} to claim.",
    "Dear customer, you have won Rs.{amount} in our monthly lucky draw. Reply WIN to claim.",
    # Financial phishing
    "URGENT: Your bank account has been SUSPENDED. Verify immediately at {url} to avoid closure.",
    "ALERT: Unauthorized access on your account. Secure it now at {url}",
    "Your KYC is pending. Failure to update within 24 hours will suspend your account. Click {url}",
    "Dear user, your account is blocked. Update PAN at {url} within 12 hours to unlock.",
    "IMPORTANT: Your credit card has been flagged. Call {phone} immediately to prevent suspension.",
    # Work from home scams
    "Earn Rs.{amount} daily working from home! No experience needed. Free registration. Call {phone}",
    "Make Rs.{amount} per week online! Part time work from home. WhatsApp {phone} now!",
    "JOB OFFER: Earn Rs.{amount} monthly from home. No target. Free registration. Apply at {url}",
    "Data entry work from home. Earn Rs.{amount}/hour. No investment. Contact {phone}",
    # Loan scams
    "Pre-approved personal loan of Rs.{amount} lakhs. Instant disbursal. No documents. Call {phone}",
    "Loan approved! Rs.{amount} transferred to your account. Accept at {url} before 6 PM today.",
    "Special offer: Home loan at 4% interest. No processing fee. Apply at {url}. Limited time!",
    # Health scams
    "LOSE {n} kg in {n2} days with our proven formula. Order now at {url}. Limited stock!",
    "Doctor approved: Lose {n} kg in {n2} weeks. No exercise needed. Order {url}",
    "FREE health checkup worth Rs.{amount}. Book now at {url}. Offer valid today only.",
    # Investment scams
    "Invest Rs.{amount} and get Rs.{amount2} in 7 days! Guaranteed returns. WhatsApp {phone}",
    "Earn 30% monthly returns on crypto investment. Start with just Rs.{amount}. Contact {phone}",
    "Stock market tips: 100% profit guaranteed. Subscribe now at {url}. Call {phone}",
]


def _rand(lo, hi):
    return str(random.randint(lo, hi))


def fill(template: str) -> str:
    vals = {
        "{email}":    random.choice(["user@gmail.com", "hello@outlook.com"]),
        "{pct}":      _rand(70, 95),
        "{service}":  random.choice(["Netflix", "Spotify", "Adobe", "LinkedIn Premium",
                                     "iCloud+", "YouTube Premium", "Hotstar"]),
        "{date}":     random.choice(["10 Oct 2026", "1 Nov 2026", "15 Dec 2026",
                                     "5 Jan 2027", "20 Oct 2026"]),
        "{name}":     random.choice(["Priya", "Rahul", "Sneha", "Arjun",
                                     "Ananya", "Vikram", "Meera", "Karthik"]),
        "{channel}":  random.choice(["#general", "#engineering", "#product",
                                     "Dev Team", "#design"]),
        "{title}":    random.choice(["Q4 Budget Review", "Sprint Planning",
                                     "Product Roadmap", "Design Review"]),
        "{n}":        _rand(100000, 999999),
        "{n2}":       _rand(2, 8),
        "{amount}":   _rand(100, 9999),
        "{amount2}":  _rand(1000, 50000),
        "{balance}":  _rand(5000, 99999),
        "{last4}":    _rand(1000, 9999),
        "{merchant}": random.choice(["Amazon", "Swiggy", "Flipkart", "BigBasket",
                                     "DMart", "Zomato", "Paytm"]),
        "{ref}":      "REF" + _rand(100000, 999999),
        "{order_id}": "OD" + _rand(10000000, 99999999),
        "{product}":  random.choice(["iPhone 15", "Samsung 4K TV", "MacBook Air",
                                     "Boat Earphones", "Nike Shoes"]),
        "{courier}":  random.choice(["BlueDart", "Delhivery", "DTDC", "Xpressbees"]),
        "{restaurant}": random.choice(["Pizza Hut", "KFC", "Dominos",
                                       "Biryani Blues", "Haldirams"]),
        "{min}":      _rand(25, 45),
        "{user}":     random.choice(["u/techbro99", "u/IndiaNews", "TechUser42",
                                     "@devcoder", "@startup_in"]),
        "{subreddit}": random.choice(["india", "learnprogramming", "Chennai",
                                      "cscareerquestions", "tamil"]),
        "{company}":  random.choice(["Google", "TCS", "Infosys", "Zomato",
                                     "Razorpay", "Freshworks", "BYJU's"]),
        "{role}":     random.choice(["Software Engineer", "Data Scientist",
                                     "Product Manager", "Frontend Developer",
                                     "DevOps Engineer"]),
        "{city}":     random.choice(["Bengaluru", "Chennai", "Hyderabad",
                                     "Mumbai", "Pune", "Delhi"]),
        "{city2}":    random.choice(["Coimbatore", "Kochi", "Mangaluru", "Mysuru"]),
        "{course}":   random.choice(["Machine Learning", "Web Development",
                                     "Data Science with Python", "DevOps Basics"]),
        "{train}":    "TN" + _rand(100, 999),
        "{flight}":   random.choice(["6E", "AI", "SG", "UK"]) + _rand(100, 999),
        "{time}":     random.choice(["10:00 AM", "2:00 PM", "11:30 AM", "3:30 PM"]),
        "{prize}":    random.choice(["iPhone 15 Pro", "Samsung Galaxy S24",
                                     "Rs.10,000 gift card", "MacBook Pro"]),
        "{url}":      random.choice(["http://win-now.xyz/claim", "http://verify-bank.tk",
                                     "http://free-prize.ml", "http://secure-kyc-update.xyz"]),
        "{phone}":    "9" + "".join([str(random.randint(0, 9)) for _ in range(9)]),
        "{year}":     "2025-26",
        "{month}":    random.choice(["September 2026", "October 2026", "August 2026"]),
        "{doc}":      random.choice(["Driving License", "PAN Card", "Degree Certificate"]),
        "{authority}": random.choice(["CBSE", "Anna University", "UIDAI", "RTO"]),
        "{repo}":     random.choice(["spam-detector", "my-api", "web-app"]),
        "{workflow}": random.choice(["CI/CD", "Run Tests", "Deploy to Prod"]),
        "{workflow}": random.choice(["CI/CD", "Run Tests", "Deploy to Prod"]),
        "{channel}":  random.choice(["#general", "#engineering", "dev-team"]),
        "{atm}":      random.choice(["SBI ATM, Chennai", "HDFC ATM, Bengaluru"]),
        "{doc}":      random.choice(["Driving License", "PAN Card"]),
    }
    result = template
    for key, val in vals.items():
        result = result.replace(key, val)
    return result


# ---------------------------------------------------------------------------
# 2.  Generate rows
# ---------------------------------------------------------------------------
print("Generating synthetic training examples...")
synthetic = []

for _ in range(2500):   # 2500 modern HAM
    synthetic.append({"text": fill(random.choice(HAM_TEMPLATES)), "label": 0})

for _ in range(500):    # 500 modern SPAM (boosts minority class)
    synthetic.append({"text": fill(random.choice(SPAM_TEMPLATES)), "label": 1})

ham_ct  = sum(1 for r in synthetic if r["label"] == 0)
spam_ct = sum(1 for r in synthetic if r["label"] == 1)
print(f"  Synthetic HAM : {ham_ct}")
print(f"  Synthetic SPAM: {spam_ct}")


# ---------------------------------------------------------------------------
# 3.  Load original UCI dataset
# ---------------------------------------------------------------------------
original_path = RAW_DIR / "spam.csv"
original = []
print(f"\nLoading original UCI dataset from {original_path} ...")
with open(original_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        txt = row.get("text", "").strip()
        lbl = int(row.get("label", 0))
        if txt:
            original.append({"text": txt, "label": lbl})

print(f"  UCI: {len(original)} rows  "
      f"(SPAM={sum(1 for r in original if r['label']==1)}, "
      f"HAM={sum(1 for r in original if r['label']==0)})")


# ---------------------------------------------------------------------------
# 4.  SpamAssassin public corpus (download if possible)
# ---------------------------------------------------------------------------
SA_ARCHIVES = [
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
     "easy_ham", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
     "hard_ham", 0),
    ("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
     "spam_sa", 1),
]

sa_dir = RAW_DIR / "spamassassin"
sa_dir.mkdir(exist_ok=True)
sa_rows = []

print("\nDownloading SpamAssassin public corpus ...")
for url, name, label in SA_ARCHIVES:
    tar_path    = sa_dir / f"{name}.tar.bz2"
    extract_dir = sa_dir / name

    if not tar_path.exists():
        try:
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, tar_path)
            print(f"  Saved  {tar_path.stat().st_size // 1024} KB")
        except Exception as exc:
            print(f"  SKIP  {name}: {exc}")
            continue

    if not extract_dir.exists():
        try:
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(sa_dir)
            print(f"  Extracted → {extract_dir}")
        except Exception as exc:
            print(f"  SKIP  {name} extraction: {exc}")
            continue

    count = 0
    for fp in extract_dir.rglob("*"):
        if not fp.is_file() or fp.name == "cmds":
            continue
        try:
            raw  = fp.read_bytes()
            text = raw.decode("latin-1", errors="replace")
            # Separate headers from body
            body = text.split("\n\n", 1)[1] if "\n\n" in text else text
            body = body[:800].strip()
            if len(body) > 30:
                sa_rows.append({"text": body, "label": label})
                count += 1
        except Exception:
            pass
    print(f"  Parsed {count} {name} emails")

sa_ham  = sum(1 for r in sa_rows if r["label"] == 0)
sa_spam = sum(1 for r in sa_rows if r["label"] == 1)
print(f"  SpamAssassin total: {len(sa_rows)}  (HAM={sa_ham}, SPAM={sa_spam})")


# ---------------------------------------------------------------------------
# 5.  Merge, deduplicate, shuffle, save
# ---------------------------------------------------------------------------
all_rows = original + synthetic + sa_rows
print(f"\nPre-dedup total : {len(all_rows)}")

seen, deduped = set(), []
for row in all_rows:
    key = row["text"].strip().lower()[:200]
    if key not in seen:
        seen.add(key)
        deduped.append(row)

random.shuffle(deduped)

total     = len(deduped)
final_ham  = sum(1 for r in deduped if r["label"] == 0)
final_spam = sum(1 for r in deduped if r["label"] == 1)

print(f"Post-dedup total: {total}")
print(f"  HAM : {final_ham:>5}  ({final_ham/total*100:.1f}%)")
print(f"  SPAM: {final_spam:>5}  ({final_spam/total*100:.1f}%)")

out_path = RAW_DIR / "spam_augmented.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["text", "label"])
    w.writeheader()
    w.writerows(deduped)

print(f"\n✅  Saved → {out_path}")
print(f"\nNext step — retrain:")
print(f'  .venv\\Scripts\\python -m spam_detector.train --csv data/raw/spam_augmented.csv --epochs 3')
