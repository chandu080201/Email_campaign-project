# Email Campaign Optimization Model Implementation Guide

This guide provides detailed instructions for implementing the email campaign optimization model in a production environment. It covers data preparation, model deployment, integration with email marketing platforms, A/B testing, and monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Model Deployment](#model-deployment)
3. [Data Pipeline](#data-pipeline)
4. [Integration with Email Marketing Platforms](#integration-with-email-marketing-platforms)
5. [A/B Testing](#ab-testing)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before implementing the model, ensure you have:

- Python 3.7+ environment
- Required packages:
  ```
  numpy>=1.20.0
  pandas>=1.3.0
  scikit-learn>=1.0.0
  imbalanced-learn>=0.8.0
  ```
- Access to user data (purchase history, country)
- Email marketing platform with API access
- Storage for model artifacts

## Model Deployment

### Option 1: Containerized Deployment

1. **Create a Docker container**:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY optimized_model.pkl .
COPY prediction_service.py .

EXPOSE 8000

CMD ["python", "prediction_service.py"]
```

2. **Create a prediction service**:

```python
# prediction_service.py
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
with open('optimized_model.pkl', 'rb') as f:
    model_info = pickle.load(f)

model = model_info['model']
threshold = model_info['threshold']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    processed_df = add_derived_features(df)
    
    # Make prediction
    prob = model.predict_proba(processed_df)[0, 1]
    prediction = 1 if prob >= threshold else 0
    
    return jsonify({
        'probability': float(prob),
        'prediction': int(prediction),
        'threshold': float(threshold)
    })

def add_derived_features(df):
    # Implement feature engineering
    # ...
    return processed_df

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

3. **Build and deploy the container**:

```bash
docker build -t email-campaign-model .
docker run -p 8000:8000 email-campaign-model
```

### Option 2: Serverless Deployment

1. **Create a serverless function** (AWS Lambda example):

```python
import json
import pickle
import pandas as pd
import numpy as np
import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Download model from S3
model_bucket = os.environ['MODEL_BUCKET']
model_key = os.environ['MODEL_KEY']
local_model_path = '/tmp/optimized_model.pkl'
s3.download_file(model_bucket, model_key, local_model_path)

# Load the model
with open(local_model_path, 'rb') as f:
    model_info = pickle.load(f)

model = model_info['model']
threshold = model_info['threshold']

def lambda_handler(event, context):
    # Parse input
    body = json.loads(event['body'])
    
    # Convert to DataFrame
    df = pd.DataFrame([body])
    
    # Apply feature engineering
    processed_df = add_derived_features(df)
    
    # Make prediction
    prob = model.predict_proba(processed_df)[0, 1]
    prediction = 1 if prob >= threshold else 0
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'probability': float(prob),
            'prediction': int(prediction),
            'threshold': float(threshold)
        })
    }

def add_derived_features(df):
    # Implement feature engineering
    # ...
    return processed_df
```

2. **Deploy using AWS SAM or Terraform**

## Data Pipeline

### Batch Processing

For batch processing of email campaigns:

1. **Extract user data**:
```python
def extract_user_data(database_connection):
    query = """
    SELECT user_id, country, past_purchases
    FROM users
    WHERE is_active = 1
    """
    return pd.read_sql(query, database_connection)
```

2. **Generate email parameters**:
```python
def generate_email_parameters(user_data, model, threshold):
    # Apply feature engineering
    processed_data = add_derived_features(user_data)
    
    # Make predictions
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    # Determine optimal parameters
    results = []
    for i, user in user_data.iterrows():
        # Find best parameters for this user
        best_params = find_best_parameters(user, model)
        
        results.append({
            'user_id': user['user_id'],
            'email_text': best_params['email_text'],
            'email_version': best_params['email_version'],
            'hour': best_params['hour'],
            'weekday': best_params['weekday'],
            'predicted_probability': best_params['probability']
        })
    
    return pd.DataFrame(results)
```

3. **Find best parameters for each user**:
```python
def find_best_parameters(user, model):
    best_prob = 0
    best_params = None
    
    # Test different combinations
    for email_text in ['short_email', 'long_email']:
        for email_version in ['personalized', 'generic']:
            for hour in [9, 10, 11, 12, 13, 14, 15, 16]:
                for weekday in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    # Create test data
                    test_data = {
                        'user_country': user['country'],
                        'user_past_purchases': user['past_purchases'],
                        'email_text': email_text,
                        'email_version': email_version,
                        'hour': hour,
                        'weekday': weekday
                    }
                    
                    # Process data
                    processed_data = add_derived_features(pd.DataFrame([test_data]))
                    
                    # Predict
                    prob = model.predict_proba(processed_data)[0, 1]
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_params = {
                            'email_text': email_text,
                            'email_version': email_version,
                            'hour': hour,
                            'weekday': weekday,
                            'probability': prob
                        }
    
    return best_params
```

### Real-time Processing

For real-time optimization:

1. **Create an API endpoint**:
```python
@app.route('/optimize_email', methods=['POST'])
def optimize_email():
    user_data = request.json
    
    # Find best parameters
    best_params = find_best_parameters_realtime(user_data, model)
    
    return jsonify(best_params)
```

2. **Implement caching for performance**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_optimal_parameters(user_country, user_past_purchases):
    # Calculate optimal parameters
    # ...
    return optimal_params
```

## Integration with Email Marketing Platforms

### Mailchimp Integration

```python
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError

def send_optimized_emails(campaign_parameters):
    client = MailchimpMarketing.Client()
    client.set_config({
        "api_key": "your-api-key",
        "server": "your-server-prefix"
    })
    
    for _, params in campaign_parameters.iterrows():
        try:
            # Create campaign
            campaign = client.campaigns.create({
                "type": "regular",
                "recipients": {
                    "list_id": "your-list-id",
                    "segment_opts": {
                        "match": "all",
                        "conditions": [{
                            "field": "email",
                            "op": "is",
                            "value": params['user_email']
                        }]
                    }
                },
                "settings": {
                    "subject_line": "New Feature Announcement",
                    "from_name": "Your Company",
                    "reply_to": "support@yourcompany.com",
                    "template_id": get_template_id(params)
                }
            })
            
            # Schedule campaign
            schedule_time = get_schedule_time(params['weekday'], params['hour'])
            client.campaigns.schedule(campaign["id"], {
                "schedule_time": schedule_time
            })
            
        except ApiClientError as error:
            print(f"Error: {error.text}")
```

### Custom Email Service Integration

```python
def integrate_with_custom_service(campaign_parameters, api_endpoint, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for _, params in campaign_parameters.iterrows():
        payload = {
            'recipient': params['user_email'],
            'template': get_template_name(params),
            'scheduled_time': get_schedule_time(params['weekday'], params['hour']),
            'personalization': {
                'user_name': params['user_name'] if params['email_version'] == 'personalized' else None
            }
        }
        
        response = requests.post(api_endpoint, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"Error sending email: {response.text}")
```

## A/B Testing

### Test Design

1. **Split users into control and test groups**:
```python
def create_test_groups(users, test_fraction=0.5):
    # Randomly assign users to groups
    users['test_group'] = np.random.choice(
        ['control', 'test'], 
        size=len(users), 
        p=[1-test_fraction, test_fraction]
    )
    
    return users
```

2. **Apply different strategies to each group**:
```python
def apply_campaign_strategies(users, model):
    # Control group: random parameters
    control_users = users[users['test_group'] == 'control'].copy()
    control_users['email_text'] = np.random.choice(['short_email', 'long_email'], size=len(control_users))
    control_users['email_version'] = np.random.choice(['personalized', 'generic'], size=len(control_users))
    control_users['hour'] = np.random.randint(8, 18, size=len(control_users))
    control_users['weekday'] = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], size=len(control_users))
    
    # Test group: optimized parameters
    test_users = users[users['test_group'] == 'test'].copy()
    optimized_params = generate_email_parameters(test_users, model, threshold)
    test_users = test_users.merge(optimized_params, on='user_id')
    
    return pd.concat([control_users, test_users])
```

3. **Analyze results**:
```python
def analyze_ab_test_results(campaign_results):
    # Calculate metrics by group
    control_metrics = calculate_metrics(campaign_results[campaign_results['test_group'] == 'control'])
    test_metrics = calculate_metrics(campaign_results[campaign_results['test_group'] == 'test'])
    
    # Calculate improvement
    improvement = {
        'absolute': test_metrics['ctr'] - control_metrics['ctr'],
        'relative': (test_metrics['ctr'] - control_metrics['ctr']) / control_metrics['ctr'] * 100
    }
    
    # Statistical significance
    contingency_table = [
        [control_metrics['clicks'], control_metrics['total'] - control_metrics['clicks']],
        [test_metrics['clicks'], test_metrics['total'] - test_metrics['clicks']]
    ]
    
    chi2, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
    
    return {
        'control': control_metrics,
        'test': test_metrics,
        'improvement': improvement,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## Monitoring and Maintenance

### Performance Monitoring

1. **Track key metrics**:
```python
def track_model_performance(predictions, actual_outcomes):
    # Calculate metrics
    precision = precision_score(actual_outcomes, predictions)
    recall = recall_score(actual_outcomes, predictions)
    f1 = f1_score(actual_outcomes, predictions)
    
    # Log metrics
    log_metrics({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'timestamp': datetime.now().isoformat()
    })
```

2. **Set up alerts**:
```python
def check_model_drift(current_metrics, baseline_metrics, threshold=0.1):
    # Calculate drift
    f1_drift = abs(current_metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics['f1_score']
    
    if f1_drift > threshold:
        send_alert(f"Model drift detected: F1 score changed by {f1_drift*100:.2f}%")
        return True
    
    return False
```

### Model Retraining

1. **Scheduled retraining**:
```python
def retrain_model_schedule():
    # Set up a scheduled job (e.g., using cron or Airflow)
    # that runs this function monthly
    
    # Load new data
    new_data = load_new_campaign_data()
    
    # Retrain model
    new_model = train_model(new_data)
    
    # Evaluate new model
    new_metrics = evaluate_model(new_model, test_data)
    
    # If new model is better, deploy it
    if new_metrics['f1_score'] > current_metrics['f1_score']:
        deploy_model(new_model)
        update_baseline_metrics(new_metrics)
```

2. **Event-based retraining**:
```python
def retrain_model_on_drift(drift_threshold=0.1):
    # Check for drift
    current_metrics = get_current_metrics()
    baseline_metrics = get_baseline_metrics()
    
    if check_model_drift(current_metrics, baseline_metrics, drift_threshold):
        # Retrain model
        new_data = load_new_campaign_data()
        new_model = train_model(new_data)
        
        # Evaluate and deploy
        new_metrics = evaluate_model(new_model, test_data)
        deploy_model(new_model)
        update_baseline_metrics(new_metrics)
```

## Troubleshooting

### Common Issues and Solutions

1. **Missing Features Error**:
   - **Issue**: `KeyError: 'feature_name'`
   - **Solution**: Ensure all required features are present in the input data and that feature engineering is applied correctly.

2. **Model Loading Error**:
   - **Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'optimized_model.pkl'`
   - **Solution**: Check that the model file path is correct and the file exists.

3. **Preprocessing Error**:
   - **Issue**: `ValueError: Input contains NaN, infinity or a value too large for dtype('float64').`
   - **Solution**: Add data validation and cleaning steps to handle missing or invalid values.

4. **Memory Error**:
   - **Issue**: `MemoryError`
   - **Solution**: Process data in batches or optimize memory usage.

### Debugging Tips

1. **Logging**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_service.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def predict_with_logging(data):
    try:
        logger.info(f"Received prediction request: {data}")
        
        # Process data
        processed_data = add_derived_features(data)
        logger.debug(f"Processed data: {processed_data}")
        
        # Make prediction
        prediction = model.predict(processed_data)
        logger.info(f"Prediction result: {prediction}")
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise
```

2. **Input Validation**:
```python
def validate_input(data):
    required_fields = ['email_text', 'email_version', 'hour', 'weekday', 'user_country', 'user_past_purchases']
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate field types
    if not isinstance(data['hour'], (int, float)) or not (0 <= data['hour'] < 24):
        raise ValueError(f"Invalid hour value: {data['hour']}")
    
    if not isinstance(data['user_past_purchases'], (int, float)) or data['user_past_purchases'] < 0:
        raise ValueError(f"Invalid user_past_purchases value: {data['user_past_purchases']}")
    
    if data['email_text'] not in ['short_email', 'long_email']:
        raise ValueError(f"Invalid email_text value: {data['email_text']}")
    
    if data['email_version'] not in ['personalized', 'generic']:
        raise ValueError(f"Invalid email_version value: {data['email_version']}")
    
    if data['weekday'] not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        raise ValueError(f"Invalid weekday value: {data['weekday']}")
    
    return True
```

## Conclusion

This implementation guide provides a comprehensive framework for deploying the email campaign optimization model in a production environment. By following these guidelines, you can effectively integrate the model with your email marketing platform, conduct A/B testing to validate its effectiveness, and set up monitoring and maintenance procedures to ensure continued performance.

For any questions or issues, please contact the model development team.
