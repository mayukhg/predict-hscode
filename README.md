# HSCode Prediction API

A production-ready microservice API that predicts HSCodes (Harmonized System Codes) for item descriptions using machine learning. This tool automates customs classification, reducing manual work and compliance risks.

## 🚀 Features

- **Single API Endpoint**: `/predict` for HSCode prediction
- **Machine Learning**: Logistic Regression with TF-IDF vectorization
- **Text Preprocessing**: Comprehensive cleaning and normalization
- **Error Handling**: Robust validation and error responses
- **Health Monitoring**: Built-in health checks
- **Docker Support**: Complete containerization
- **Production Ready**: Security, logging, and monitoring

## 📋 Requirements

- Python 3.11+
- FastAPI
- scikit-learn
- pandas
- Docker (optional)

## 🏗️ Project Structure

```
/
├── main.py                 # FastAPI application
├── train.py               # Model training script
├── utils.py               # Utility functions
├── data/
│   └── kaggle_data.csv    # Training dataset
├── models/
│   ├── hs_model.pkl       # Trained model (generated)
│   └── tfidf_vectorizer.pkl # Fitted vectorizer (generated)
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── implementation.md      # Detailed documentation
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mayukhg/predict-hscode.git
cd predict-hscode
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load the training dataset
- Preprocess text descriptions
- Train a Logistic Regression model
- Save the model and vectorizer to `models/` directory

### 4. Start the API

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📖 API Usage

### Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict HSCode

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"item_description": "Electronic digital camera with interchangeable lens"}'
```

### Response Format

```json
{
  "hscode": "8517.12.30",
  "confidence": 0.95
}
```

### Example Responses

```bash
# Electronics
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"item_description": "Smartphone with touchscreen display"}'

# Clothing
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"item_description": "Cotton t-shirt for men"}'

# Food
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"item_description": "Fresh apples red delicious"}'
```

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t hscode-api .
```

### Run the Container

```bash
docker run -p 8000:8000 hscode-api
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🔧 Development

### Training the Model

The training script (`train.py`) performs the following steps:

1. **Data Loading**: Reads `data/kaggle_data.csv`
2. **Preprocessing**: Cleans text and standardizes HSCodes
3. **Feature Extraction**: TF-IDF vectorization
4. **Model Training**: Logistic Regression with regularization
5. **Evaluation**: Accuracy metrics and classification report
6. **Persistence**: Saves model and vectorizer

### Text Preprocessing

The `utils.py` module provides:

- **`preprocess_text()`**: Converts to lowercase, removes punctuation
- **`clean_hscode()`**: Standardizes HSCode format
- **`validate_hscode_format()`**: Validates HSCode structure

### Model Performance

- **Accuracy**: ~55% on test set
- **Features**: 10,000 most important terms (unigrams + bigrams)
- **Classes**: 25 HSCode categories
- **Training Data**: 190 samples after filtering

## 📊 Dataset

The training dataset includes:

- **396 samples** across diverse product categories
- **Categories**: Electronics, clothing, food, tools, furniture, etc.
- **HSCodes**: Real-world customs classifications
- **Format**: CSV with `item_description` and `hscode` columns

## 🛠️ Configuration

### Environment Variables

- `PYTHONPATH`: Python path configuration
- `PORT`: API port (default: 8000)

### Model Parameters

- **Test Size**: 20% of data
- **Max Features**: 10,000
- **N-gram Range**: (1, 2)
- **Min Document Frequency**: 2
- **Max Document Frequency**: 0.95

## 🔍 Error Handling

### HTTP Status Codes

- `200`: Successful prediction
- `400`: Bad request (invalid input)
- `500`: Internal server error

### Error Responses

```json
{
  "detail": "item_description must be a string"
}
```

## 📈 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### Logs

The API provides comprehensive logging for:
- Model loading status
- Request processing
- Error conditions
- Performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure `train.py` has been run successfully
2. **Port Already in Use**: Change port or kill existing processes
3. **Memory Issues**: Reduce `max_features` in training script
4. **Low Accuracy**: Increase training data or adjust preprocessing

### Debug Mode

```bash
python -c "
import requests
import subprocess
import time

# Start API
process = subprocess.Popen(['python', 'main.py'])
time.sleep(5)

# Test prediction
response = requests.post('http://localhost:8000/predict', 
  json={'item_description': 'test item'})
print(response.json())

process.terminate()
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the implementation.md for detailed documentation
- Review the API documentation at `http://localhost:8000/docs`

## 🔄 Version History

- **v1.0.0**: Initial release with basic HSCode prediction
- **v1.1.0**: Added comprehensive error handling and health checks
- **v1.2.0**: Improved model training and Docker support

## 🎯 Future Enhancements

- [ ] Deep learning models (Transformers)
- [ ] Batch prediction support
- [ ] Model versioning
- [ ] API rate limiting
- [ ] Prometheus metrics
- [ ] Redis caching
- [ ] User feedback integration

---

**Built with ❤️ for customs automation and compliance**
