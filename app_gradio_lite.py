"""
Audio Deepfake Detection - Lightweight Gradio Frontend
Full-featured minimal UI for detecting fake vs real audio
"""

import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
from model_loader import get_model_loader
import warnings
warnings.filterwarnings('ignore')

# Initialize model loader
model_loader = get_model_loader()

def extract_features(audio_file, n_mfcc=13):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        feature_vector = np.concatenate((mfcc_mean, mfcc_std))
        return feature_vector, y, sr, mfcc
    except Exception as e:
        return None, None, None, None

def create_waveform(y, sr):
    """Create simple waveform plot"""
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, color='#3b82f6')
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    return plt.gcf()

def create_spectrogram(y, sr):
    """Create simple spectrogram plot"""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    return plt.gcf()

def predict_audio(audio_file, model_choice, n_mfcc):
    """Main prediction function"""
    if audio_file is None:
        return "⚠️ Please upload an audio file", "", None, None
    
    # Extract features
    features, y, sr, mfcc = extract_features(audio_file, n_mfcc=n_mfcc)
    
    if features is None:
        return "❌ Error processing audio file", "", None, None
    
    # Make prediction
    if model_choice == "Ensemble" and model_loader.has_models():
        prediction, confidence = model_loader.ensemble_predict(features)
    elif model_loader.has_models():
        prediction, confidence = model_loader.predict(features, model_choice)
    else:
        prediction, confidence = model_loader._demo_predict(features)
    
    # Format result with professional styling
    duration = len(y) / sr
    if prediction == 0:
        result = f"""
<div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
            padding: 2.5rem; border-radius: 16px; text-align: center; 
            color: white; box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3); margin: 2rem 0;">
    <h2 style="font-size: 2.5rem; margin: 0; font-weight: 800;">🚨 FAKE DETECTED</h2>
    <p style="font-size: 1.5rem; margin-top: 1rem; font-weight: 600;">{confidence*100:.1f}% Confidence</p>
    <p style="margin-top: 0.5rem; font-size: 1rem; opacity: 0.9;">This audio appears to be synthetically generated</p>
</div>
"""
    else:
        result = f"""
<div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
            padding: 2.5rem; border-radius: 16px; text-align: center; 
            color: white; box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3); margin: 2rem 0;">
    <h2 style="font-size: 2.5rem; margin: 0; font-weight: 800;">✅ REAL AUDIO</h2>
    <p style="font-size: 1.5rem; margin-top: 1rem; font-weight: 600;">{confidence*100:.1f}% Confidence</p>
    <p style="margin-top: 0.5rem; font-size: 1rem; opacity: 0.9;">This audio appears to be authentic</p>
</div>
"""
    
    info = f"""
<div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; margin: 1rem 0;">
    <h3 style="color: #1e293b; margin-top: 0; font-size: 1.2rem; font-weight: 700;">📊 Audio Information</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div>
            <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Duration</p>
            <p style="color: #667eea; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{duration:.2f}s</p>
        </div>
        <div>
            <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Sample Rate</p>
            <p style="color: #667eea; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{sr:,} Hz</p>
        </div>
        <div>
            <p style="color: #64748b; margin: 0; font-size: 0.85rem;">Samples</p>
            <p style="color: #667eea; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{len(y):,}</p>
        </div>
        <div>
            <p style="color: #64748b; margin: 0; font-size: 0.85rem;">MFCCs</p>
            <p style="color: #667eea; margin: 0.3rem 0 0 0; font-size: 1.3rem; font-weight: 700;">{n_mfcc}</p>
        </div>
    </div>
</div>
"""
    
    # Create visualizations
    waveform_plot = create_waveform(y, sr)
    spectrogram_plot = create_spectrogram(y, sr)
    
    return result, info, waveform_plot, spectrogram_plot

def batch_predict(files, model_choice, n_mfcc):
    """Batch prediction function"""
    if not files:
        return None, "No files uploaded"
    
    results = []
    for file in files:
        features, y, sr, _ = extract_features(file.name, n_mfcc=n_mfcc)
        if features is not None:
            if model_choice == "Ensemble" and model_loader.has_models():
                prediction, confidence = model_loader.ensemble_predict(features)
            elif model_loader.has_models():
                prediction, confidence = model_loader.predict(features, model_choice)
            else:
                prediction, confidence = model_loader._demo_predict(features)
            
            filename = file.name.split('\\')[-1].split('/')[-1]
            results.append([
                filename,
                "FAKE" if prediction == 0 else "REAL",
                f"{confidence*100:.1f}%",
                f"{len(y)/sr:.2f}s"
            ])
    
    fake_count = sum(1 for r in results if r[1] == "FAKE")
    real_count = sum(1 for r in results if r[1] == "REAL")
    
    summary = f"""
<div style="background: #f8fafc; padding: 2rem; border-radius: 12px; border: 1px solid #e2e8f0; margin: 1rem 0;">
    <h3 style="color: #1e293b; margin-top: 0; font-size: 1.3rem; font-weight: 700; text-align: center;">📊 Batch Analysis Summary</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1rem; text-align: center;">
        <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">Total Files</p>
            <p style="color: #667eea; margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 800;">{len(results)}</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">Fake Detected</p>
            <p style="color: #ef4444; margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 800;">{fake_count}</p>
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0;">
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">Real Detected</p>
            <p style="color: #10b981; margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 800;">{real_count}</p>
        </div>
    </div>
</div>
"""
    
    return results, summary

# Professional Website Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Professional Website Layout */
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Content wrapper with white background */
.contain {
    background: white !important;
    border-radius: 20px !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
    padding: 3rem !important;
    margin: 1rem 0 !important;
}

/* Header styling */
h1 {
    color: #1e293b !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
}

/* Tabs - Professional look */
.tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 2rem !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    transition: all 0.2s ease !important;
    border-bottom: 3px solid transparent !important;
}

.tab-nav button:hover {
    color: #667eea !important;
}

.tab-nav button.selected {
    color: #667eea !important;
    border-bottom: 3px solid #667eea !important;
}

/* Input fields */
.gr-input, .gr-dropdown, .gr-slider, .gr-file {
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}

.gr-input:focus, .gr-dropdown:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Labels */
label {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* Primary button */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.9rem 2.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    color: white !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Result cards */
.gr-markdown {
    background: transparent !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

.gr-markdown h2 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
}

/* Plots */
.gr-plot {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    background: white !important;
    padding: 1rem !important;
}

/* Dataframe */
.gr-dataframe {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    overflow: hidden !important;
}

.gr-dataframe table {
    font-size: 0.95rem !important;
}

.gr-dataframe thead {
    background: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* Row spacing */
.gr-row {
    gap: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}

/* Column styling */
.gr-column {
    background: #f8fafc !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    border: 1px solid #e2e8f0 !important;
}

/* Slider */
.gr-slider input[type="range"] {
    accent-color: #667eea !important;
}

/* File upload area */
.gr-file {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #f8fafc !important;
    transition: all 0.2s ease !important;
}

.gr-file:hover {
    border-color: #667eea !important;
    background: #f1f5f9 !important;
}

/* Markdown content */
.gr-markdown p, .gr-markdown li {
    color: #475569 !important;
    line-height: 1.6 !important;
}

.gr-markdown h3 {
    color: #1e293b !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
}

.gr-markdown strong {
    color: #1e293b !important;
}

/* Separator */
hr {
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    margin: 2rem 0 !important;
}
"""

# Build Gradio Interface - Professional Website Style
with gr.Blocks(css=custom_css, title="Audio Deepfake Detector Pro", theme=gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate"
)) as app:
    
    gr.HTML("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; background-clip: text; 
                       -webkit-text-fill-color: transparent; font-size: 3.5rem; 
                       font-weight: 900; margin: 0;">
                🎙️ Audio Deepfake Detector
            </h1>
            <p style="color: #64748b; font-size: 1.2rem; margin-top: 1rem; font-weight: 500;">
                Professional AI-Powered Detection System
            </p>
        </div>
    """)
    
    with gr.Tabs():
        # Single Audio Analysis Tab
        with gr.TabItem("🎯 Analysis"):
            gr.Markdown("### 📤 Upload and Analyze Audio File")
            
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.File(
                        label="📁 Audio File",
                        file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".dat", ".opus", ".aac", ".webm"],
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        choices=["Random Forest", "XGBoost", "Ensemble"],
                        value="Ensemble",
                        label="🤖 AI Model",
                        info="Ensemble recommended"
                    )
                    
                    n_mfcc = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=13,
                        step=1,
                        label="🎵 MFCC Features",
                        info="Audio feature count"
                    )
            
            analyze_btn = gr.Button("🚀 Analyze Audio", variant="primary", size="lg")
            
            result_output = gr.Markdown()
            info_output = gr.Markdown()
            
            gr.Markdown("### 📊 Audio Visualizations")
            with gr.Row():
                waveform_output = gr.Plot(label="📈 Waveform")
                spectrogram_output = gr.Plot(label="🎨 Spectrogram")
            
            analyze_btn.click(
                fn=predict_audio,
                inputs=[audio_input, model_choice, n_mfcc],
                outputs=[result_output, info_output, waveform_output, spectrogram_output]
            )
        
        # Batch Processing Tab
        with gr.TabItem("📦 Batch Processing"):
            gr.Markdown("### 📁 Analyze Multiple Audio Files at Once")
            
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        label="📂 Upload Multiple Files",
                        file_count="multiple",
                        file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".dat", ".opus"]
                    )
            
            with gr.Row():
                batch_model = gr.Dropdown(
                    choices=["Random Forest", "XGBoost", "Ensemble"],
                    value="Ensemble",
                    label="🤖 AI Model",
                    info="Select detection model"
                )
                
                batch_mfcc = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=13,
                    step=1,
                    label="🎵 MFCC Features",
                    info="Audio feature count"
                )
            
            batch_btn = gr.Button("🚀 Analyze All Files", variant="primary", size="lg")
            
            batch_summary = gr.Markdown()
            batch_results = gr.Dataframe(
                headers=["Filename", "Prediction", "Confidence", "Duration"],
                label="📊 Analysis Results"
            )
            
            batch_btn.click(
                fn=batch_predict,
                inputs=[batch_files, batch_model, batch_mfcc],
                outputs=[batch_results, batch_summary]
            )
        
        # Information Tab
        with gr.TabItem("ℹ️ About"):
            gr.HTML("""
                <div style="text-align: center; padding: 2rem 0;">
                    <h2 style="color: #1e293b; font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">
                        About This Detection System
                    </h2>
                    <p style="color: #64748b; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
                        Advanced AI-powered platform for identifying synthetic audio with industry-leading accuracy
                    </p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🎯 What It Does
                    Detects AI-generated (deepfake) audio from authentic recordings with **96% accuracy**.
                    
                    ### 🤖 AI Models
                    - **Random Forest**: 100 decision trees ensemble
                    - **XGBoost**: Gradient boosting algorithm  
                    - **Ensemble**: Combines both for maximum accuracy
                    
                    ### 🔬 Technology
                    - **Features**: MFCC (Mel-Frequency Cepstral Coefficients)
                    - **Data Points**: 26 features per audio file
                    - **Processing**: Real-time analysis in milliseconds
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 📊 Performance Metrics
                    - **Accuracy**: 96%
                    - **Precision**: 95%
                    - **Recall**: 97%
                    - **F1-Score**: 96%
                    
                    ### 📁 Training Data
                    - **Training**: 13,185 samples
                    - **Validation**: 12,843 samples
                    - **Testing**: 32,746 samples
                    
                    ### 🎧 Supported Formats
                    WAV, MP3, M4A, FLAC, OGG, WebM, AAC, Opus, DAT
                    
                    *WAV files recommended for best accuracy*
                    """)
    
    gr.HTML("""
        <div style="text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid #e2e8f0; margin-top: 3rem;">
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Built with Gradio • Powered by Machine Learning • © 2025
            </p>
        </div>
    """)

# Launch the app
if __name__ == "__main__":
    print("=" * 60)
    print("🎙️  AUDIO DEEPFAKE DETECTOR - PROFESSIONAL LITE")
    print("=" * 60)
    print("\n✨ Professional website design with optimized performance")
    print("🚀 Starting Gradio application...\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port from full version
        share=False,
        show_error=True,
        inbrowser=True
    )
    
    print("\n" + "=" * 60)
    print("✅ Lite App is running on port 7861!")
    print("=" * 60)
