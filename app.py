import gradio as gr
import joblib
import json

# Load the trained model and vectorizer
with open("career_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = joblib.load(vectorizer_file)

# Load career descriptions from the JSON file
with open("assets/career_descriptions.json", "r") as file:
    career_descriptions = json.load(file)

# Function to predict career based on qualification and skills
def predict_career(qualification, skills):
    try:
        input_text = qualification.strip() + " " + skills.strip()  # Combine the input
        input_vector = vectorizer.transform([input_text])  # Transform input
        prediction = model.predict(input_vector)[0]  # Predict the career
        prediction_proba = model.predict_proba(input_vector).max()  # Get confidence score

        description = career_descriptions.get(prediction, "Description not available.")  # Get career description
        result = f"🎯 **Predicted Career:** {prediction}\n\n📝 **Description:** {description}\n\n📊 **Confidence:** {prediction_proba:.2%}"
        return result
    except Exception as e:
        return f"❌ An error occurred during prediction: {str(e)}"

# Gradio UI layout
iface = gr.Interface(
    fn=predict_career,  # Prediction function
    inputs=[
        gr.Textbox(label="Enter Qualification", placeholder="e.g., B.Tech Computer Science"),
        gr.Textbox(label="Enter Skills", placeholder="e.g., Python, Java, Machine Learning")
    ],
    outputs="text",  # Output as text
    title="Smart Career Counselor",  # App title
    description="Predict the best career based on qualification and skills.",  # Description
    live=True  # Enable live prediction
)

# Launch the app
iface.launch()
