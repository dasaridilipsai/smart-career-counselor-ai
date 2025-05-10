import gradio as gr
import joblib
import json

# Load the trained model and vectorizer
with open("career_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

# Load career descriptions and suggested courses
with open("assets/career_descriptions.json", "r", encoding="utf-8") as file:
    career_descriptions = json.load(file)

# Function to predict career based on qualification and skills
def predict_career(qualification, skills):
    try:
        input_text = qualification.strip() + " " + skills.strip()  # Combine inputs
        input_vector = vectorizer.transform([input_text])           # Vectorize input
        prediction = model.predict(input_vector)[0]                 # Predict career
        prediction_proba = model.predict_proba(input_vector).max()  # Get top confidence

        # Get career description and course suggestions
        info = career_descriptions.get(prediction, {})
        description = info.get("description", "No description available.")
        courses = info.get("courses", ["No course suggestions available."])
        course_list = "\n".join([f"• {c}" for c in courses])

        # Final result string
        result = (
            f"🎯 **Predicted Career:** {prediction}\n\n"
            f"📝 **Description:** {description}\n\n"
            f"📊 **Confidence:** {prediction_proba:.2%}\n\n"
            f"🎓 **Suggested Courses:**\n{course_list}"
        )
        return result

    except Exception as e:
        return f"❌ An error occurred during prediction: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=predict_career,
    inputs=[
        gr.Textbox(label="Enter Qualification", placeholder="e.g., B.Tech Computer Science"),
        gr.Textbox(label="Enter Skills", placeholder="e.g., Python, SQL, React JS")
    ],
    outputs="markdown",
    title="Smart Career Counselor 🎓",
    description="Enter your qualification and skills to get personalized career guidance powered by AI.",
    live=True
)

# Launch the app with a public URL
iface.launch(share=True)
