# 🎓 Smart Career Counselor AI Project

An AI-powered career recommendation system that suggests the most suitable career paths based on a user's qualifications, skills, and interests. Built using Python, Scikit-learn, and Gradio.

---

## 🚀 Demo

Coming soon on Hugging Face Spaces!

---

## 📌 Features

- ✅ Predicts Top 3 Career Options
- ✅ Confidence Scores in Percentages
- ✅ Career Descriptions
- ✅ Suggested Courses for Skill Building
- ✅ Clean and Interactive Gradio UI

---

## 🧠 How It Works

The system uses a `RandomForestClassifier` trained on a curated dataset of career paths. Inputs are processed using `TfidfVectorizer`, and the model returns the top career predictions with explanations.

---

## 📂 Folder Structure

smart-career-counselor/
│
├── app.py # Gradio frontend + prediction logic
├── train_model.py # Model training script
├── career_data.csv # Raw dataset
├── career_data_cleaned.csv # Cleaned dataset
├── career_model.pkl # Trained model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── assets/
│ └── career_descriptions.json # Career info with descriptions & courses


---

## 🧪 Test Cases

| Input Skills                                   | Predicted Career         |
|------------------------------------------------|---------------------------|
| Python, SQL, Excel                             | Data Analyst              |
| HTML, CSS, JavaScript, React JS                | Web Developer             |
| Machine Learning, Python, Deep Learning        | AI Engineer               |
| Tally, Accounting, Excel                       | Accountant                |
| Communication, Sales, Management               | Business Analyst          |

---

## ⚙️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/dasaridilipsai/career-counselor-ai-project.git
cd career-counselor-ai-project


Install requirements
pip install -r requirements.txt
Launch the app
python app.py


 Requirements
Python 3.8+
Gradio
Scikit-learn
Pandas
Joblib
NumPy

Author
Dasari Dilip Sai
LinkedIn • GitHub

