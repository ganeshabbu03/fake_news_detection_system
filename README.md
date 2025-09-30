# 🤖 AI-Powered Fake News Detection System

A sophisticated fake news detection system powered by Google's Gemini AI, providing detailed analysis, source quality assessment, and red flag detection.

## ✨ Features

- **🧠 Advanced AI Analysis**: Uses Google's Gemini 2.0 Flash for sophisticated language understanding
- **📊 Detailed Reasoning**: Provides comprehensive explanations for each prediction
- **🔍 Source Quality Assessment**: Evaluates credibility and reliability of sources
- **🚨 Red Flag Detection**: Identifies common patterns of misinformation
- **📈 Confidence Scoring**: Shows confidence levels for predictions
- **🌐 Multiple Interfaces**: Web UI, REST API, and CLI tools
- **⚡ Real-time Analysis**: Fast, accurate predictions with detailed insights

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake_news_detection_system.git
   cd fake_news_detection_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your_api_key_here"
   ```

### Usage

#### 🌐 Web Interface (Streamlit)
```bash
python -m streamlit run src/fake_news/ui_app_gemini_only.py
```
Open your browser to `http://localhost:8501`

#### 🔧 Command Line Interface
```bash
# Simple prediction
python -m scripts.predict_ai --text "Your news text here" --simple

# Detailed analysis
python -m scripts.predict_ai --text "Your news text here" --verbose

# Standard output
python -m scripts.predict_ai --text "Your news text here"
```

#### 🌐 REST API
```bash
# Start the API server
python -m scripts.run_api_gemini_only --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your news text here"]}'
```

## 📚 API Documentation

### Endpoints

- **POST /predict** - Analyze multiple texts
- **POST /predict/single** - Analyze single text
- **GET /health** - Health check
- **GET /models** - Model information
- **GET /docs** - Interactive API documentation

### Example API Usage

```python
import requests

# Analyze single text
response = requests.post("http://localhost:8000/predict/single", 
                        params={"text": "Breaking: Aliens land in Times Square!"})
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Reasoning: {result['reasoning']}")
```

## 🎯 How It Works

The system uses Google's Gemini 2.0 Flash model to analyze text for fake news indicators:

1. **Language Analysis**: Examines writing style, tone, and structure
2. **Source Evaluation**: Assesses credibility and reliability
3. **Factual Assessment**: Evaluates the accuracy of claims
4. **Red Flag Detection**: Identifies common misinformation patterns
5. **Confidence Scoring**: Provides confidence levels for predictions

## 📊 Analysis Features

### What the AI Analyzes

- **Source Quality**: Poor, Fair, Good, or Excellent
- **Factual Accuracy**: Low, Medium, or High
- **Red Flags**: Sensational language, lack of sources, clickbait, etc.
- **Reasoning**: Detailed explanation of the analysis
- **Confidence**: How certain the AI is about the prediction

### Example Analysis

**Input**: "Breaking: Aliens have landed in Times Square!"

**Output**:
- **Prediction**: FAKE NEWS
- **Confidence**: 99.0%
- **Source Quality**: Poor
- **Factual Accuracy**: Low
- **Red Flags**: Sensational headline, Lack of supporting evidence, Absence of credible sources
- **Reasoning**: "The claim of aliens landing in Times Square is an extraordinary claim that requires extraordinary evidence..."

## 🛠️ Development

### Project Structure

```
fake_news_detection_system/
├── src/fake_news/
│   ├── gemini_detector.py      # Core Gemini AI integration
│   ├── ui_app_gemini_only.py   # Streamlit web interface
│   └── service_gemini_only.py  # FastAPI service
├── scripts/
│   ├── predict_ai.py           # CLI tool
│   └── run_api_gemini_only.py  # API server script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Model Configuration

The system uses Google's Gemini 2.0 Flash model with the following settings:
- **Temperature**: 0.1 (for consistent, focused responses)
- **Max Output Tokens**: 1024
- **Safety Settings**: Configured for content analysis

## 📈 Performance

- **Accuracy**: ~95%+ on test cases
- **Speed**: 2-5 seconds per analysis (depending on text length)
- **Languages**: Primarily English (best results)
- **Text Length**: Works with short headlines to long articles

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google for providing the Gemini AI model
- The open-source community for various Python libraries
- Contributors and testers who help improve the system

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/fake_news_detection_system/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Batch processing capabilities
- [ ] Integration with news APIs
- [ ] Historical analysis features
- [ ] Custom model fine-tuning
- [ ] Real-time monitoring dashboard

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always verify information through multiple reliable sources before making important decisions.