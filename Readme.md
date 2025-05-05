# Start new training
python main.py --algorithm sfac

# Resume from checkpoint
python main.py --algorithm sfac --checkpoint checkpoints/SFAC_Breakout-v4_20250501-120000_ep100.pt

# Train both algorithms (one from checkpoint)
python main.py --algorithm both --checkpoint checkpoints/SFAC_Breakout-v4_20250501-120000_ep100.pt