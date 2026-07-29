import gradio as gr
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("obesity_knn_model.pkl")

# Mappings
gender_map = {
    "Male": 0,
    "Female": 1
}

yes_no_map = {
    "no": 0,
    "yes": 1
}

calc_map = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

caec_map = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

mtrans_map = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
}

# Reverse mapping for output
obesity_map = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}


def predict(age, gender, height, weight, calc, scc, caec, favc,
            fcvc, ncp, smoke, ch2o, family_history, faf, tue, mtrans):

    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_map[gender],
        "Height": height,
        "Weight": weight,
        "CALC": calc_map[calc],
        "SCC": yes_no_map[scc],
        "CAEC": caec_map[caec],
        "FAVC": yes_no_map[favc],
        "FCVC": fcvc,
        "NCP": ncp,
        "SMOKE": yes_no_map[smoke],
        "CH2O": ch2o,
        "family_history_with_overweight": yes_no_map[family_history],
        "FAF": faf,
        "TUE": tue,
        "MTRANS": mtrans_map[mtrans]
    }])

    prediction = model.predict(data)[0]

    return obesity_map[int(prediction)]


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Height (meters)"),
        gr.Number(label="Weight (kg)"),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="CALC"),
        gr.Dropdown(["no", "yes"], label="SCC"),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"], label="CAEC"),
        gr.Dropdown(["no", "yes"], label="FAVC"),
        gr.Number(label="FCVC"),
        gr.Number(label="NCP"),
        gr.Dropdown(["no", "yes"], label="SMOKE"),
        gr.Number(label="CH2O"),
        gr.Dropdown(["no", "yes"], label="Family History with Overweight"),
        gr.Number(label="FAF"),
        gr.Number(label="TUE"),
        gr.Dropdown(
            [
                "Public_Transportation",
                "Walking",
                "Automobile",
                "Motorbike",
                "Bike"
            ],
            label="MTRANS"
        )
    ],
    outputs=gr.Textbox(label="Predicted Obesity Level"),
    title="Obesity Prediction Using K-Nearest Neighbors (KNN)",
    description="""
### Developed By
**Name:** Palak Madan

**College:** Panipat Institute of Engineering and Technology (PIET), Samalkha, Panipat

**Branch:** Computer Science and Engineering (CSE)

**Roll No.:** 28240290

Predict the obesity level using the K-Nearest Neighbors (KNN) Machine Learning algorithm.
"""
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)