# hybrid_model_debugger.py
import pickle
import numpy as np
import sys
import traceback

def debug_model(model_path, test_text):
    """
    Debugs the hybrid model by running a detailed test prediction and inspecting the outputs
    at each stage of the process
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Load model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        print(f"Model loaded successfully. Type: {type(model_data)}")
        
        # Determine the type of model
        if isinstance(model_data, dict):
            print("\nModel is a dictionary with keys:")
            for key in model_data:
                print(f"  - {key} ({type(model_data[key])})")
            
            # Look for classifier in the dictionary
            classifier = None
            if 'model' in model_data:
                classifier = model_data['model']
                print("Using 'model' key as classifier")
            elif 'classifier' in model_data:
                classifier = model_data['classifier']
                print("Using 'classifier' key as classifier")
            else:
                # Try to find a component with predict method
                for key, component in model_data.items():
                    if hasattr(component, 'predict'):
                        classifier = component
                        print(f"Using '{key}' as classifier (has predict method)")
                        break
        else:
            # Direct classifier
            classifier = model_data
            print("Model is a direct classifier object")
        
        if not classifier:
            print("ERROR: Could not identify a classifier component in the model")
            return
        
        # Check for mlb
        mlb = None
        if hasattr(classifier, 'mlb'):
            mlb = classifier.mlb
            print("\nFound MultiLabelBinarizer on classifier")
            if hasattr(mlb, 'classes_'):
                print(f"Available classes: {mlb.classes_}")
            else:
                print("WARNING: MultiLabelBinarizer has no classes_ attribute")
        else:
            print("\nNo MultiLabelBinarizer found on classifier")
            
            # Check if mlb is in the dictionary
            if isinstance(model_data, dict) and 'mlb' in model_data:
                mlb = model_data['mlb']
                print("Found MultiLabelBinarizer in model dictionary")
                if hasattr(mlb, 'classes_'):
                    print(f"Available classes: {mlb.classes_}")
                else:
                    print("WARNING: MultiLabelBinarizer has no classes_ attribute")
        
        # Check for alpha parameter
        alpha = getattr(classifier, 'alpha', None)
        print(f"\nAlpha parameter: {alpha}")
        
        # Check for threshold parameter
        threshold = getattr(classifier, 'threshold', None)
        print(f"Threshold parameter: {threshold}")
        
        # Try making a prediction
        print(f"\nTesting prediction with text: '{test_text}'")
        
        # Try different prediction approaches
        approaches = [
            ("Standard prediction with text as list", lambda: classifier.predict([test_text])),
            ("With specific alpha and threshold", lambda: classifier.predict([test_text], alpha=0.6, threshold=0.4)),
            ("With return_scores=True", lambda: classifier.predict([test_text], return_scores=True)),
            ("All parameters", lambda: classifier.predict([test_text], alpha=0.6, threshold=0.4, return_scores=True))
        ]
        
        for description, predict_func in approaches:
            print(f"\n--- {description} ---")
            try:
                result = predict_func()
                print(f"Result type: {type(result)}")
                print(f"Result value: {result}")
                
                # If it's a numpy array, try to interpret it
                if isinstance(result, np.ndarray):
                    print(f"Array shape: {result.shape}")
                    print(f"Array contents: {result}")
                    
                    if mlb and hasattr(mlb, 'classes_'):
                        try:
                            # Check if it's a binary array
                            if len(result.shape) == 2:  # First dim is samples, second is classes
                                labels = mlb.classes_[result[0].astype(bool)].tolist()
                                print(f"Converted to labels: {labels}")
                        except Exception as e:
                            print(f"Error converting to labels: {e}")
                
                # If it's a list, check the first item
                elif isinstance(result, list) and len(result) > 0:
                    print(f"First item type: {type(result[0])}")
                    print(f"First item value: {result[0]}")
                    
                # If it's a dictionary, check its structure
                elif isinstance(result, dict):
                    print("Dictionary keys:")
                    for key in result:
                        value = result[key]
                        print(f"  - {key} ({type(value)})")
                        
                        # Show a sample of the value
                        if isinstance(value, (list, tuple)) and len(value) > 0:
                            print(f"    Sample: {value[:3]}...")
                        elif isinstance(value, dict) and len(value) > 0:
                            sample_keys = list(value.keys())[:3]
                            print(f"    Sample keys: {sample_keys}...")
                        else:
                            print(f"    Value: {value}")
            
            except Exception as e:
                print(f"Error during prediction: {e}")
                print(traceback.format_exc())
        
        print("\nDebugging complete")
        
    except Exception as e:
        print(f"Error loading or processing model: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    model_path = r"C:\Users\tueyc\CMKL Year 1\nomad_sync_app\backend\hybrid_interest_classifier.pkl"
    test_text = "I hike mountains and explore cultures while traveling. I also love cooking new recipes."
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_text = sys.argv[2]
    
    debug_model(model_path, test_text)