# Customer Support Ticket Analyzer

A Streamlit demo from the AI-Powered Customer Support Analytics portfolio project. Enter a hypothetical support ticket, choose a priority and channel, and see how the project's models read it. It is a portfolio demonstration, not a production tool.

This repository is a minimal deployment bundle for Streamlit Community Cloud. It contains only the app, the shared inference code, and the model files the app needs.

## What the demo shows

The Streamlit app (`streamlit_app.py`) calls the project's shared inference layer (`support_analytics/inference.py`) in the same process:

- **Sentiment:** a pretrained, general-purpose binary sentiment model (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`, DistilBERT fine-tuned on SST-2 movie-review sentences). It was not trained on customer-support tickets. Its confidence is confidence in the sentiment label, not a measure of customer dissatisfaction.
- **Topic:** a TF-IDF + Logistic Regression classifier that stands in for the project's BERTopic topic discovery. It agrees with the BERTopic assignments on about 65.7% of tickets in its held-out split. Its scores are relative model confidence, not calibrated probabilities, and the app shows the top 3 candidate topics. Topic labels are human-interpreted, exploratory labels with review status "proposed", not a validated issue taxonomy; some topics are marked as noise clusters.
- **Experimental model score:** an experimental model-derived score from a Gradient Boosting model, shown only in a collapsed section. Its evaluation found no demonstrated predictive discrimination (held-out ROC-AUC 0.508, 95% CI 0.461–0.556). It is not a validated probability, not confirmed dissatisfaction, and must not be used for prioritization or triage decisions.

**Observed dissatisfaction** is a separate, CSAT-grounded metric: 39.80% of the 2,769 tickets with an observed CSAT rating had a rating of 2 or lower. It is a historical rate from the dataset, not a property of the entered ticket, and it is not the same thing as the experimental model score.

The app accepts up to 5,000 characters of ticket text and includes four hand-written example tickets.

## Running

Streamlit Community Cloud settings:

- Main file: `streamlit_app.py`
- Python version: 3.11 (select it under Advanced settings; scikit-learn 1.9.0 is required to load the pickled models)
- Dependencies: `requirements.txt` (CPU-only PyTorch)
- No secrets are required.

To run locally with Python 3.11:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py --server.address localhost
```

The sentiment model is downloaded from the Hugging Face Hub the first time a ticket is analyzed. All models are then cached for the life of the Streamlit server process.

## Privacy

Do not enter personal or confidential information. The app does not store submitted ticket text, model inference runs within the application, and Streamlit usage statistics are disabled. The hosting platform may retain standard request/application logs.

The deployed demo does not include the source ticket dataset or any ticket descriptions. It includes the app code, topic labels, aggregate evaluation metrics, and two trained model files. One of them, the topic classifier, contains a learned vocabulary of 9,529 words and two-word phrases, each of which appeared in at least three tickets in its training data. That vocabulary includes some common personal names and web-address fragments, but no complete email addresses or phone numbers. The full project repository, which is not part of this deployment bundle, has not been privacy-sanitized and may contain third-party identifiers or contact details.
