# Run main.py

## Installation & Running

```bash
# Create and activate virtual environment
python -m venv torch-env  
source torch-env/bin/activate  

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium gymnasium[classic-control] numpy

# Run the script
python main.py
```

## Dependencies
- torch
- torchvision
- torchaudio
- gymnasium
- numpy