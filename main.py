import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox
import sqlite3
from tkinter import ttk

# importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',
                      delimiter='\t', quoting=3)
corpus = []
rras_code = "Wyd^H3R"
food_rev = {}
food_perc = {}

conn = sqlite3.connect('Restaurant_food_data.db')
c = conn.cursor()

# Pre-processing the dataset
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    
    review = ' '.join(review)
    corpus.append(review)

# Feature extraction & Train/test split
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

variables = []
clr_variables = []

# Initialize food items and database
foods = ["Idly", "Dosa", "Vada", "Roti", "Meals", "Veg Biryani",
         "Egg Biryani", "Chicken Biryani", "Mutton Biryani",
         "Ice Cream", "Noodles", "Manchooriya", "Orange juice",
         "Apple Juice", "Pineapple juice", "Banana juice"]

# Initialize food review and percentage dictionaries
for i in foods:
    food_rev[i] = []
    food_perc[i] = [0.0, 0.0]

# Create database table if not exists
def init_database():
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS item 
                 (Item_name text, No_of_customers text,
                  No_of_positive_reviews text, No_of_negative_reviews text,
                  Positive_percentage text, Negative_percentage text)''')
    
    # Check if table is empty
    c.execute("SELECT COUNT(*) FROM item")
    if c.fetchone()[0] == 0:
        for food in foods:
            c.execute("INSERT INTO item VALUES (?,?,?,?,?,?)",
                     (food, "0", "0", "0", "0.0%", "0.0%"))
    conn.commit()
    conn.close()

# Initialize the database
init_database()

# Main application window
def create_main_window():
    root = Tk()
    root.title("Restaurant Review Analysis System")
    
    # Configure window
    try:
        root.state('zoomed')
    except Exception:
        root.attributes("-zoomed", True)
    
    # Main title
    title = Label(root, text="RESTAURANT REVIEW ANALYSIS SYSTEM",
                 font=('Arial', 24, 'bold', 'underline'))
    title.pack(pady=20)
    
    # User type selection
    user_frame = Frame(root)
    user_frame.pack(pady=30)
    
    Label(user_frame, text="Select User Type:",
          font=('Arial', 16)).pack(pady=10)
    
    Button(user_frame, text="Customer", font=('Arial', 14),
           width=20, command=lambda: customer_window(root)).pack(pady=10)
    
    Button(user_frame, text="Owner", font=('Arial', 14),
           width=20, command=lambda: owner_window(root)).pack(pady=10)
    
    return root

# Customer review window
def customer_window(parent):
    window = Toplevel(parent)
    window.title("Customer Review")
    
    try:
        window.state('zoomed')
    except Exception:
        window.attributes("-zoomed", True)
    
    # Title
    Label(window, text="Submit Your Review",
          font=('Arial', 20, 'bold')).pack(pady=20)
    
    # Food selection frame
    food_frame = LabelFrame(window, text="Select Food Items",
                           font=('Arial', 12))
    food_frame.pack(padx=20, pady=10, fill='x')
    
    # Create checkboxes in a grid
    food_vars = {}
    for i, food in enumerate(foods):
        var = IntVar()
        food_vars[food] = var
        Checkbutton(food_frame, text=food, variable=var,
                   font=('Arial', 10)).grid(row=i//4, column=i%4,
                                          padx=10, pady=5, sticky='w')
    
    # Rating frame
    rating_frame = LabelFrame(window, text="Rate Your Experience",
                            font=('Arial', 12))
    rating_frame.pack(padx=20, pady=10, fill='x')
    
    rating_var = IntVar(value=0)
    rating_label = Label(rating_frame, text="Select Rating:",
                        font=('Arial', 10))
    rating_label.pack(pady=5)
    
    # Create star rating buttons
    stars_frame = Frame(rating_frame)
    stars_frame.pack(pady=5)
    
    def update_rating(rating):
        rating_var.set(rating)
        # Update star colors
        for i in range(5):
            if i < rating:
                star_buttons[i].config(text="★", fg="gold")
            else:
                star_buttons[i].config(text="☆", fg="gray")
    
    star_buttons = []
    for i in range(5):
        btn = Button(stars_frame, text="☆", font=('Arial', 20),
                    fg="gray", command=lambda x=i+1: update_rating(x))
        btn.pack(side=LEFT, padx=2)
        star_buttons.append(btn)
    
    # Review entry
    review_frame = LabelFrame(window, text="Your Review",
                             font=('Arial', 12))
    review_frame.pack(padx=20, pady=10, fill='x')
    
    review_text = Text(review_frame, height=5, width=50,
                      font=('Arial', 10))
    review_text.pack(padx=10, pady=10)
    
    # Submit button
    def submit_review():
        selected_foods = [food for food, var in food_vars.items()
                         if var.get() == 1]
        review = review_text.get("1.0", END).strip()
        rating = rating_var.get()
        
        if not selected_foods:
            messagebox.showerror("Error", "Please select at least one food item!")
            return
        
        if not review:
            messagebox.showerror("Error", "Please enter your review!")
            return
            
        if rating == 0:
            messagebox.showerror("Error", "Please select a rating!")
            return
        
        # Process review
        processed_review = process_review(review)
        sentiment = predict_sentiment(processed_review)
        
        # Update database with rating
        update_database(selected_foods, sentiment, rating)
        
        messagebox.showinfo("Success", "Thank you for your review!")
        window.destroy()
    
    Button(window, text="Submit Review", font=('Arial', 12),
           command=submit_review).pack(pady=20)

# Owner window
def owner_window(parent):
    window = Toplevel(parent)
    window.title("Owner Login")
    
    try:
        window.state('zoomed')
    except Exception:
        window.attributes("-zoomed", True)
    
    # Title
    Label(window, text="Owner Login",
          font=('Arial', 20, 'bold')).pack(pady=20)
    
    # Code entry
    code_frame = Frame(window)
    code_frame.pack(pady=20)
    
    Label(code_frame, text="Enter RRAS Code:",
          font=('Arial', 12)).pack(side='left', padx=5)
    
    code_entry = Entry(code_frame, show="*", font=('Arial', 12))
    code_entry.pack(side='left', padx=5)
    
    def verify_code():
        if code_entry.get() == rras_code:
            show_owner_dashboard(window)
        else:
            messagebox.showerror("Error", "Invalid RRAS code!")
    
    Button(window, text="Login", font=('Arial', 12),
           command=verify_code).pack(pady=10)

# Owner dashboard
def show_owner_dashboard(parent):
    window = Toplevel(parent)
    window.title("Owner Dashboard")
    
    try:
        window.state('zoomed')
    except Exception:
        window.attributes("-zoomed", True)
    
    # Title
    Label(window, text="Owner Dashboard",
          font=('Arial', 20, 'bold')).pack(pady=20)
    
    # Buttons
    Button(window, text="View Statistics",
           font=('Arial', 12), width=20,
           command=lambda: show_statistics(window)).pack(pady=10)
    
    Button(window, text="Clear Item Data",
           font=('Arial', 12), width=20,
           command=lambda: clear_item_data(window)).pack(pady=10)
    
    Button(window, text="Clear All Data",
           font=('Arial', 12), width=20,
           command=lambda: clear_all_data(window)).pack(pady=10)

# Helper functions
def process_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    review = [ps.stem(word) for word in review
              if word not in set(all_stopwords)]
    
    return ' '.join(review)

def predict_sentiment(review):
    X = cv.transform([review]).toarray()
    prediction = classifier.predict(X)[0]
    
    # Handle negation
    if "not" in review:
        prediction = 1 - prediction
    
    return prediction

def update_database(selected_foods, sentiment, rating):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    
    # Add rating column if it doesn't exist
    try:
        c.execute("ALTER TABLE item ADD COLUMN Average_rating REAL DEFAULT 0")
        c.execute("ALTER TABLE item ADD COLUMN Total_ratings INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    for food in selected_foods:
        c.execute("SELECT * FROM item WHERE Item_name=?", (food,))
        record = c.fetchone()
        
        customers = int(record[1]) + 1
        positives = int(record[2]) + (1 if sentiment == 1 else 0)
        negatives = int(record[3]) + (1 if sentiment == 0 else 0)
        
        # Update rating
        current_avg = float(record[6]) if len(record) > 6 else 0
        total_ratings = int(record[7]) if len(record) > 7 else 0
        new_total = total_ratings + 1
        new_avg = ((current_avg * total_ratings) + rating) / new_total
        
        pos_percent = f"{(positives/customers)*100:.1f}%"
        neg_percent = f"{(negatives/customers)*100:.1f}%"
        
        c.execute("""UPDATE item SET 
                    No_of_customers=?, No_of_positive_reviews=?,
                    No_of_negative_reviews=?, Positive_percentage=?,
                    Negative_percentage=?, Average_rating=?,
                    Total_ratings=? WHERE Item_name=?""",
                 (str(customers), str(positives), str(negatives),
                  pos_percent, neg_percent, f"{new_avg:.1f}",
                  str(new_total), food))
    
    conn.commit()
    conn.close()

def show_statistics(parent):
    window = Toplevel(parent)
    window.title("Review Statistics")
    window.configure(bg='#f0f0f0')  # Light gray background
    
    try:
        window.state('zoomed')
    except Exception:
        window.attributes("-zoomed", True)
    
    # Title with better styling
    title_frame = Frame(window, bg='#f0f0f0')
    title_frame.pack(pady=20)
    
    Label(title_frame, text="Review Statistics Dashboard",
          font=('Arial', 24, 'bold'),
          bg='#f0f0f0',
          fg='#2c3e50').pack()
    
    # Create main container
    main_frame = Frame(window, bg='#f0f0f0')
    main_frame.pack(padx=20, pady=10, fill=BOTH, expand=True)
    
    # Create Treeview with better styling
    style = ttk.Style()
    style.configure("Custom.Treeview",
                   background="#ffffff",
                   foreground="#2c3e50",
                   rowheight=30,
                   fieldbackground="#ffffff")
    style.configure("Custom.Treeview.Heading",
                   font=('Arial', 12, 'bold'),
                   background="#3498db",
                   foreground="white")
    style.map("Custom.Treeview.Heading",
              background=[('active', '#2980b9')])
    
    # Define columns with simpler names
    columns = {
        "Food Item": 150,
        "Total Reviews": 100,
        "Rating": 100,
        "Satisfaction": 120
    }
    
    tree = ttk.Treeview(main_frame, columns=list(columns.keys()),
                        show='headings', style="Custom.Treeview")
    
    # Configure columns
    for col, width in columns.items():
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor='center')
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Fetch and process data
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM item")
    records = c.fetchall()
    
    # Insert data with simplified information
    for record in records:
        # Calculate satisfaction percentage
        total_reviews = int(record[1])  # No_of_customers
        positive_reviews = int(record[2])  # No_of_positive_reviews
        satisfaction = f"{(positive_reviews/total_reviews)*100:.1f}%" if total_reviews > 0 else "0%"
        
        # Format rating with stars
        avg_rating = float(record[6]) if len(record) > 6 else 0
        rating_stars = "★" * int(avg_rating) + "☆" * (5 - int(avg_rating))
        rating_display = f"{rating_stars} ({avg_rating:.1f})"
        
        # Insert simplified data
        tree.insert('', 'end', values=(
            record[0],  # Food Item
            total_reviews,  # Total Reviews
            rating_display,  # Rating with stars
            satisfaction  # Satisfaction percentage
        ))
    
    # Pack the treeview and scrollbar
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    # Add summary frame
    summary_frame = Frame(window, bg='#f0f0f0')
    summary_frame.pack(pady=20, fill=X)
    
    # Calculate overall statistics
    total_items = len(records)
    total_reviews = sum(int(record[1]) for record in records)
    avg_rating = sum(float(record[6]) if len(record) > 6 else 0 for record in records) / total_items if total_items > 0 else 0
    
    # Create summary labels with better styling
    summary_style = {'font': ('Arial', 12), 'bg': '#f0f0f0', 'fg': '#2c3e50'}
    
    Label(summary_frame, text=f"Total Food Items: {total_items}",
          **summary_style).pack(side=LEFT, padx=20)
    Label(summary_frame, text=f"Total Reviews: {total_reviews}",
          **summary_style).pack(side=LEFT, padx=20)
    Label(summary_frame, text=f"Average Rating: {avg_rating:.1f} ★",
          **summary_style).pack(side=LEFT, padx=20)
    
    # Add exit button with better styling
    Button(window, text="Close Dashboard",
           font=('Arial', 12),
           bg='#e74c3c',
           fg='white',
           padx=20,
           pady=10,
           command=window.destroy).pack(pady=20)
    
    conn.close()

# Clear item data
def clear_item_data(parent):
    window = Toplevel(parent)
    window.title("Clear Item Data")
    
    try:
        window.state('zoomed')
    except Exception:
        window.attributes("-zoomed", True)
    
    # Title
    Label(window, text="Select Items to Clear",
          font=('Arial', 20, 'bold')).pack(pady=20)
    
    # Create checkboxes
    food_vars = {}
    for i, food in enumerate(foods):
        var = IntVar()
        food_vars[food] = var
        Checkbutton(window, text=food, variable=var,
                   font=('Arial', 10)).grid(row=i//4, column=i%4,
                                          padx=10, pady=5, sticky='w')
    
    # Clear selected items
    def clear_selected():
        selected = [food for food, var in food_vars.items()
                   if var.get() == 1]
        
        if not selected:
            messagebox.showerror("Error", "Please select items to clear!")
            return
        
        if messagebox.askyesno("Confirm",
                              "Are you sure you want to clear selected items?"):
            conn = sqlite3.connect('Restaurant_food_data.db')
            c = conn.cursor()
            
            for food in selected:
                c.execute("""UPDATE item SET 
                           No_of_customers='0', No_of_positive_reviews='0',
                           No_of_negative_reviews='0', Positive_percentage='0.0%',
                           Negative_percentage='0.0%' WHERE Item_name=?""",
                        (food,))
            
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Selected items cleared!")
            window.destroy()
    
    Button(window, text="Clear Selected",
           font=('Arial', 12), command=clear_selected).pack(pady=20)

# Clear all data
def clear_all_data(parent):
    if messagebox.askyesno("Confirm",
                          "Are you sure you want to clear all data?"):
        conn = sqlite3.connect('Restaurant_food_data.db')
        c = conn.cursor()
        
        for food in foods:
            c.execute("""UPDATE item SET 
                       No_of_customers='0', No_of_positive_reviews='0',
                       No_of_negative_reviews='0', Positive_percentage='0.0%',
                       Negative_percentage='0.0%' WHERE Item_name=?""",
                    (food,))
        
        conn.commit()
        conn.close()
        messagebox.showinfo("Success", "All data cleared!")

# Start the application
if __name__ == "__main__":
    root = create_main_window()
    root.mainloop()

