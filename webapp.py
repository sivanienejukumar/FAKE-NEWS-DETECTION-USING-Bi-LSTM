from tensorflow.keras.models import load_model
import gradio as gr

# Load the TensorFlow model
model = load_model("new_model.h5")


# Function to make predictions
def predict_fake_news(text):
    try:
        
        prediction = model.predict([text])[0]
        
        # Return the prediction (assuming a binary classification)
        if prediction > 0.5:
            return "This news is fake."
        else:
            return "This news is real."
    except Exception as e:
        # Handle exceptions gracefully
        return "An error occurred during prediction: {}".format(str(e))

# Create a Gradio interface
gr.Interface(fn=predict_fake_news, 
             inputs="text", 
             outputs="text",
             title="Fake News Detection",
             description="Enter a news article and click submit to detect if it's fake or real.").launch()

