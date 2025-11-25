import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

def test_model():
    print("Testing DeepSeek-OCR local setup...")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
    
    try:
        # Load Model (Matches app.py)
        model_path = "deepseek-ai/DeepSeek-OCR"
        print(f"Loading model from {model_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).cuda().eval()
        
        print("SUCCESS: Model loaded successfully on GPU!")
        print(f"Model Class: {type(model).__name__}")
        print("Available methods (partial):", [m for m in dir(model) if not m.startswith("_")])
        
        # Create dummy image
        dummy_image = Image.new('RGB', (100, 100), color='white')
        dummy_path = "test_image.jpg"
        dummy_image.save(dummy_path)
        
        import inspect
        print(f"Signature of model.infer: {inspect.signature(model.infer)}")

        print("Attempting model.infer()...")
        try:
            # Correct call based on signature: (tokenizer, prompt, image_file, output_path)
            prompt = "Convert this page to markdown."
            output_dir = "./test_output"
            
            # Note: infer might return the text OR just save it. We'll check the return value.
            response = model.infer(
                tokenizer=tokenizer,
                prompt=prompt,
                image_file=dummy_path,
                output_path=output_dir,
                save_results=True # Let's try saving to see if it works
            )
            print("Inference successful!")
            print(f"Response type: {type(response)}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()

        print("Environment is ready for packing.")
        
    except Exception as e:
        print(f"FAILURE: Model load failed.\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
