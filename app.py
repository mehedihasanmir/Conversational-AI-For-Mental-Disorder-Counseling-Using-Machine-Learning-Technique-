import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)
CORS(app)

# Load model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'mental_health_model.pkl')
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load a sample of the training data for LIME
# IMPORTANT: Replace 'training_data_sample.npy' with your actual training data path
# This data should be preprocessed in the same way as the model's input.
# You need to create this file (e.g., by saving a subset of your preprocessed training data)
try:
    training_data_for_lime_path = os.path.join(os.path.dirname(__file__), 'training_data_sample.npy')
    # Check if the file exists before attempting to load
    if os.path.exists(training_data_for_lime_path):
        training_data_for_lime = np.load(training_data_for_lime_path)
        print("✅ Training data for LIME loaded successfully!")
    else:
        training_data_for_lime = None
        print(f"⚠️ Warning: training_data_sample.npy not found at {training_data_for_lime_path}. LIME explanations will not be available unless initialized with dummy data or this file is provided.")
except Exception as e:
    print(f"❌ Error loading training data for LIME: {e}")
    training_data_for_lime = None


# Feature configuration
EXPECTED_FEATURES = [
    'mood_swing', 'optimism', 'sadness', 'exhausted', 'authority_respect',
    'euphoric', 'suicidal_thoughts', 'sleep_disorder', 'sexual_activity',
    'concentration'
]

SCALE_MAP = {"Seldom": 1, "Sometimes": 2, "Usually": 3, "Most-Often": 4}
YES_NO_MAP = {"Yes": 1, "No": 0}

friendly_names = {
    'mood_swing': "Mood Swings",
    'suicidal_thoughts': "Suicidal Thoughts",
    'authority_respect': "Respect for Authority",
    'sadness': "Sadness",
    'exhausted': "Exhaustion",
    'euphoric': "Euphoric Feelings",
    'sleep_disorder': "Sleep Disorders",
    'sexual_activity': "Sexual Activity",
    'optimism': "Optimism",
    'concentration': "Concentration"
}

# Labels
label_map = {
    0: "Normal! \nHey, you’re doing great — just remember I’m here anytime you want to chat or check in 😊",
    1: "Bipolar Type-1! \nYour feelings have their ups and downs, and that’s okay — let’s take it one step at a time together 🌊",
    2: "Bipolar Type-2! \nYou’re not alone — I’m here to help you understand your emotions and support you always 🤝",
    3: "Depression! \nSome days are tough, but I’m here to listen and help you find a little light whenever you need 🌟"
}

# Create LIME explainer (initialized once with loaded training data or dummy data as fallback)
lime_explainer = None
if training_data_for_lime is not None:
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=training_data_for_lime, # Use loaded real training data
            feature_names=EXPECTED_FEATURES,
            class_names=list(label_map.values()),
            mode='classification',
            discretize_continuous=True # Set to False if your features are all discrete/categorical
        )
        print("✅ LIME Explainer initialized successfully with loaded training data")
    except Exception as e:
        print(f"❌ Error initializing LIME Explainer with loaded training data: {e}")
        lime_explainer = None
else:
    # Fallback: Initialize LIME Explainer with minimal dummy data if real data is not found
    # NOTE: Explanations might be less reliable with limited dummy data.
    print("⚠️ Initializing LIME Explainer with limited dummy data as a fallback.")
    dummy_training_data = np.array([
        [1, 6, 2, 3, 1, 2, 0, 2, 6, 7],  # Example 1
        [0, 3, 4, 4, 1, 4, 1, 4, 3, 2],  # Example 2
        [1, 8, 3, 3, 0, 3, 0, 3, 5, 6],  # Example 3
        [0, 5, 1, 1, 0, 1, 0, 1, 8, 9],  # Example 4
        [1, 7, 3, 2, 1, 2, 1, 3, 4, 5],  # Example 5
    ])
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=dummy_training_data,
            feature_names=EXPECTED_FEATURES,
            class_names=list(label_map.values()),
            mode='classification',
            discretize_continuous=True
        )
        print("✅ LIME Explainer initialized with fallback dummy data.")
    except Exception as e:
        print(f"❌ Error initializing LIME Explainer with dummy data: {e}")
        lime_explainer = None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    input_values = []
    for feature in EXPECTED_FEATURES:
        value = data.get(feature)
        if value is None:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Process input values based on their expected type (categorical/numerical mapping)
        if feature in ['mood_swing', 'authority_respect', 'suicidal_thoughts']:
            # Handle 'Yes'/'No' strings or direct 0/1 integers
            processed_value = YES_NO_MAP.get(value)
            if processed_value is None: # If not 'Yes'/'No', try converting to int (assuming 0 or 1)
                try:
                    processed_value = int(value)
                    if processed_value not in [0, 1]:
                        return jsonify({"error": f"Invalid value for {feature}: {value}. Expected 'Yes', 'No', 0, or 1."}), 400
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for {feature}: {value}. Expected 'Yes', 'No', 0, or 1."}), 400
            input_values.append(processed_value)

        elif feature in ['sadness', 'exhausted', 'euphoric', 'sleep_disorder']:
            # Handle scale strings or direct 1-4 integers
            processed_value = SCALE_MAP.get(value)
            if processed_value is None: # If not a scale string, try converting to int (assuming 1-4)
                try:
                    processed_value = int(value)
                    if processed_value not in [1, 2, 3, 4]:
                        return jsonify({"error": f"Invalid value for {feature}: {value}. Expected 'Seldom', 'Sometimes', 'Usually', 'Most-Often', or 1-4."}), 400
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid value for {feature}: {value}. Expected 'Seldom', 'Sometimes', 'Usually', 'Most-Often', or 1-4."}), 400
            input_values.append(processed_value)
        else:
            # For 'optimism' and 'concentration', assuming they are already numerical (e.g., 1-10)
            try:
                input_values.append(float(value)) # Convert to float for numerical features
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numerical value for {feature}: {value}. Expected a number."}), 400

    try:
        # Ensure the input array is a 2D array for the model
        features_array = np.array(input_values).reshape(1, -1)

        # Predict
        prediction_proba = model.predict_proba(features_array)[0]
        prediction_class_index = int(np.argmax(prediction_proba))
        disorder_label = label_map.get(prediction_class_index, "Unknown condition")

        # LIME explanation
        explanation_text = "No specific explanation available."
        top_contributions = []

        if lime_explainer is None:
            explanation_text = "LIME explainer not initialized. Cannot generate explanation."
        else:
            try:
                # LIME expects a 1D array for a single instance
                explanation = lime_explainer.explain_instance(
                    features_array[0],
                    model.predict_proba,
                    num_features=3, # Number of top features to explain
                    top_labels=1 # Explain only the top predicted label
                )

                # Get top features for the predicted class
                # explanation.as_list() returns a list of tuples: (feature_name_with_value_range, weight)
                contribs = explanation.as_list(label=prediction_class_index)
                explanation_lines = []
                for feature_str_lime, value in contribs:
                    # LIME's feature_str_lime might be like 'mood_swing=1', 'optimism > 5.0', 'sadness <= 2'
                    # We want to map this back to our friendly names
                    
                    # Find the original feature name from the LIME string
                    original_feature_name = None
                    for expected_feat in EXPECTED_FEATURES:
                        # Check if the expected feature name is part of the LIME string
                        # This simple check works for exact matches or "feature=value" forms
                        # For "feature > value" or "feature < value", you might need regex
                        if expected_feat in feature_str_lime:
                            original_feature_name = expected_feat
                            break
                    
                    display_feature_name = friendly_names.get(original_feature_name, original_feature_name if original_feature_name else feature_str_lime)
                    
                    explanation_lines.append(f"- {display_feature_name}: {feature_str_lime}")
                    top_contributions.append({
                        "feature": display_feature_name,
                        "raw_explanation": feature_str_lime, # Keep raw string for full detail
                        "importance": round(value, 4)
                    })

                if explanation_lines:
                    explanation_text = "The key features that influenced this prediction are:<br>" + "<br>".join(explanation_lines)
                else:
                    explanation_text = "No significant features found for explanation."

            except Exception as e:
                print(f"❌ LIME explanation error: {e}")
                explanation_text = f"Could not generate explanation due to LIME error: {e}"

        return jsonify({
            "prediction": disorder_label,
            "explanation": explanation_text,
            "top_features": top_contributions
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == '__main__':
    # Ensure you create 'mental_health_model.pkl' and 'training_data_sample.npy'
    # in the same directory as this app.py file before running.
    # For local development, debug=True is useful. For production, set debug=False.
    app.run(debug=True, port=5000)