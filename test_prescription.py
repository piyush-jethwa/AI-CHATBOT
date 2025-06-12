import gradio as gr
from brain_of_the_doctor import generate_prescription

def generate_prescription_ui(diagnosis, language):
    """Generate and display prescription"""
    prescription = generate_prescription(diagnosis, language)
    return prescription

with gr.Blocks() as demo:
    with gr.Row():
        diagnosis_input = gr.Textbox(label="Diagnosis")
        language_input = gr.Dropdown(
            choices=["English", "Hindi"], 
            value="English",
            label="Language"
        )
    generate_btn = gr.Button("Generate Prescription")
    prescription_output = gr.Textbox(label="Prescription")
    
    generate_btn.click(
        fn=generate_prescription_ui,
        inputs=[diagnosis_input, language_input],
        outputs=prescription_output
    )

if __name__ == "__main__":
    demo.launch()
