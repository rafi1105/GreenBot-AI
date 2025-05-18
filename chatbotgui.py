import tkinter as tk
from tkinter import scrolledtext, font, Frame, Canvas, PhotoImage
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

# NLP resources
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {"university", "please", "can"}

# Load and preprocess dataset
with open("ndata.json", "r") as f:
    data = json.load(f)


# Optimized text preprocessing function
def preprocess(text):
    text = text.lower().strip()  # Normalize to lowercase and remove extra spaces
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])  # Remove non-alphanumeric characters
    words = [lemmatizer.lemmatize(word) for word in text.split() if
             word not in stop_words]  # Lemmatization and stopword removal
    return ' '.join(words)


# Preprocess the questions and keywords
questions = [preprocess(item["question"]) for item in data]
answers = [item["answer"] for item in data]
raw_keywords = [item["keywords"] for item in data]
normalized_keywords = [[preprocess(kw) for kw in kw_list] for kw_list in raw_keywords]

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(questions)

# Clustering to find optimal number of clusters
silhouette_scores = []
cluster_range = range(2, min(20, len(data) // 10) or 3)
for n in cluster_range:
    kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, batch_size=100)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Choose the optimal cluster count
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, random_state=0, batch_size=100)
labels = kmeans.fit_predict(X)

# Map clusters to questions
cluster_to_indices = {i: [] for i in range(optimal_clusters)}
for idx, label in enumerate(labels):
    cluster_to_indices[label].append(idx)


# Caching results using lru_cache for better performance
@lru_cache(maxsize=500)
def find_best_match(user_input, threshold=0.3, keyword_boost=0.15):
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])
    predicted_cluster = kmeans.predict(user_vec)[0]
    relevant_indices = cluster_to_indices[predicted_cluster]
    similarities = cosine_similarity(user_vec, X[relevant_indices]).flatten()

    for i, idx in enumerate(relevant_indices):
        if any(kw in processed_input for kw in normalized_keywords[idx]):
            similarities[i] += keyword_boost

    best_local_idx = np.argmax(similarities)
    best_score = similarities[best_local_idx]
    best_global_idx = relevant_indices[best_local_idx]
    return answers[best_global_idx] if best_score >= threshold else "Could you clarify your question?"


# GUI Setup
def on_enter(event):
    if entry.get() == "Ask here!":
        entry.delete(0, tk.END)
        entry.config(fg='black')


def on_leave(event):
    if entry.get() == "":
        entry.insert(0, "Ask here!")
        entry.config(fg='grey')


def create_circle(canvas, x, y, r, fill_color, outline_color=None, width=1):
    """Create a circle on the canvas"""
    return canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill_color, outline=outline_color, width=width)


def draw_profile_picture(canvas, x, y, size, color, text=""):
    """Draw a profile picture circle with initials"""
    create_circle(canvas, x, y, size, color)
    if text:
        canvas.create_text(x, y, text=text, fill="white", font=("Helvetica", int(size/2), "bold"))
    return canvas


def send_message():
    user_text = entry.get()
    if user_text.strip() == "" or user_text == "Ask here!":
        return
    
    chat_window.config(state=tk.NORMAL)
    
    # Create user message with profile picture (left-aligned)
    chat_window.insert(tk.END, "\n", "user_spacing")
    
    # Insert profile picture placeholder for user
    profile_frame = Frame(chat_window, bg="#f0f0f0", height=30, width=30)
    user_canvas = Canvas(profile_frame, width=30, height=30, bg="#f0f0f0", highlightthickness=0)
    user_canvas.pack()
    draw_profile_picture(user_canvas, 15, 15, 15, "#0084ff", "U")
    chat_window.window_create(tk.END, window=profile_frame)
    
    # Insert the actual message
    chat_window.insert(tk.END, f" {user_text}", "user_bubble")
    chat_window.insert(tk.END, "\n", "user_spacing")
    
    # Get and create bot message with profile picture (right-aligned)
    response = find_best_match(user_text)
    
    # Create a new text mark to properly align the bot's profile picture to the right
    chat_window.insert(tk.END, " " * 60, "bot_bubble") # Insert spaces to push picture to the right
    
    # Insert profile picture placeholder for bot
    profile_frame_bot = Frame(chat_window, bg="#f0f0f0", height=30, width=30)
    bot_canvas = Canvas(profile_frame_bot, width=30, height=30, bg="#f0f0f0", highlightthickness=0)
    bot_canvas.pack()
    draw_profile_picture(bot_canvas, 15, 15, 15, "#8a2be2", "B")
    chat_window.window_create(tk.END, window=profile_frame_bot)
    
    # Go to next line and insert the bot response
    chat_window.insert(tk.END, "\n", "bot_spacing")
    chat_window.insert(tk.END, f"{response}", "bot_bubble")
    chat_window.insert(tk.END, "\n\n", "bot_spacing")
    
    chat_window.config(state=tk.DISABLED)
    chat_window.see(tk.END)  # Auto-scroll to the bottom
    entry.delete(0, tk.END)


# Add dashboard profiles above the chat window
def create_dashboard():
    dashboard_frame = Frame(chat_container, bg="#f0f0f0", height=50)
    dashboard_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Create user profile
    user_profile_frame = Frame(dashboard_frame, bg="#f0f0f0")
    user_profile_frame.pack(side=tk.LEFT, padx=10)
    
    user_canvas = Canvas(user_profile_frame, width=40, height=40, bg="#f0f0f0", highlightthickness=0)
    user_canvas.pack(side=tk.LEFT)
    draw_profile_picture(user_canvas, 20, 20, 20, "#0084ff", "U")
    
    user_label = tk.Label(user_profile_frame, text="You", font=("Helvetica", 10), bg="#f0f0f0")
    user_label.pack(side=tk.LEFT, padx=5)
    
    # Create bot profile
    bot_profile_frame = Frame(dashboard_frame, bg="#f0f0f0")
    bot_profile_frame.pack(side=tk.RIGHT, padx=10)
    
    bot_label = tk.Label(bot_profile_frame, text="Bot", font=("Helvetica", 10), bg="#f0f0f0")
    bot_label.pack(side=tk.RIGHT, padx=5)
    
    bot_canvas = Canvas(bot_profile_frame, width=40, height=40, bg="#f0f0f0", highlightthickness=0)
    bot_canvas.pack(side=tk.RIGHT)
    draw_profile_picture(bot_canvas, 20, 20, 20, "#8a2be2", "B")
    
    return dashboard_frame


root = tk.Tk()
root.title("Messenger Chat")
root.geometry("500x600")
root.configure(bg="#f0f0f0")  # Light grey background like messenger

# Chat container
chat_container = Frame(root, bg="#f0f0f0")
chat_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create dashboard
create_dashboard()

# Chat window to display conversation
chat_window = scrolledtext.ScrolledText(
    chat_container, 
    wrap=tk.WORD, 
    state=tk.DISABLED,
    bg="#f0f0f0",  # Light grey background
    borderwidth=0,
    highlightthickness=0,
    font=("Helvetica", 11)
)
chat_window.pack(fill=tk.BOTH, expand=True)

# Define text tags for styling message bubbles
chat_window.tag_configure(
    "user_bubble", 
    justify="left",             # Changed from right to left
    rmargin=100,                # Changed margin to accommodate left positioning
    lmargin1=20, 
    lmargin2=20,
    background="#f0f0f0",       # Same as chat background
    foreground="#0084ff",       # Blue text for user messages
    borderwidth=0,
    relief="flat",
    font=("Helvetica", 11, "bold"),
    spacing1=8,
    spacing3=15
)

chat_window.tag_configure(
    "bot_bubble", 
    justify="right",            # Changed from left to right
    rmargin=20,                 # Changed margin to accommodate right positioning
    lmargin1=100, 
    lmargin2=100,
    background="#f0f0f0",       # Same as chat background
    foreground="#8a2be2",       # Purple text for bot messages
    borderwidth=0,
    relief="flat",
    font=("Helvetica", 11, "bold"),
    spacing1=8,
    spacing3=15
)

chat_window.tag_configure("user_spacing", spacing1=8, spacing3=8)
chat_window.tag_configure("bot_spacing", spacing1=8, spacing3=8)

# Input container (with a different background for the input area)
input_container = Frame(root, bg="#ffffff", height=60)
input_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

# Text input box
entry = tk.Entry(
    input_container, 
    fg='grey', 
    font=('Helvetica', 12),
    bd=1,
    relief="solid"
)
entry.insert(0, "Ask here!")
entry.bind("<FocusIn>", on_enter)
entry.bind("<FocusOut>", on_leave)
entry.bind("<Return>", lambda event: send_message())  # Enter key as send
entry.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)

# Send button with messenger-like styling
send_button = tk.Button(
    input_container, 
    text="Send", 
    command=send_message,
    bg="#0084ff",  # Facebook messenger blue
    fg="white",
    relief="flat",
    font=("Helvetica", 10, "bold"),
    width=8,
    padx=10

)
send_button.pack(side=tk.RIGHT, padx=5, pady=10)

# Welcome message
chat_window.config(state=tk.NORMAL)

# Create a new text mark to properly align the bot's profile picture to the right
chat_window.insert(tk.END, " " * 60, "bot_bubble") # Insert spaces to push picture to the right

# Insert profile picture placeholder for bot
profile_frame_bot = Frame(chat_window, bg="#f0f0f0", height=30, width=30)
bot_canvas = Canvas(profile_frame_bot, width=30, height=30, bg="#f0f0f0", highlightthickness=0)
bot_canvas.pack()
draw_profile_picture(bot_canvas, 15, 15, 15, "#8a2be2", "B")
chat_window.window_create(tk.END, window=profile_frame_bot)

# Go to next line and insert welcome message
chat_window.insert(tk.END, "\n", "bot_spacing")
chat_window.insert(tk.END, "Welcome to the Green-chat -- Ask me any question of GUB.", "bot_bubble")
chat_window.insert(tk.END, "\n\n", "bot_spacing")
chat_window.config(state=tk.DISABLED)

root.mainloop()
