import gradio as gr
import numpy as np
from sklearn.linear_model import LinearRegression

# Model Setup
size = np.array([[800],[1000],[1200],[1500],[1800]])
price = np.array([40,50,55,70,85])
model = LinearRegression().fit(size, price)

def predict_price(sqft):
    prediction = model.predict([[sqft]])
    return f"${prediction[0]:,.2f}k"

# Gradio Interface
demo = gr.Interface(
    fn=predict_price,
    inputs=gr.Number(label="House Size (sqft)", value=1500),
    outputs=gr.Textbox(label="Predicted Price"),
    title="House Price Predictor"
)

# share=True creates a public URL automatically
demo.launch(share=True)
