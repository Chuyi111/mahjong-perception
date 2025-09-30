# Mahjong Perception (Windows)

Perception module for Mahjong Soul: captures the game window and streams frames for downstream tile recognition and state reconstruction.

## Quick start (Windows 10)

```powershell
git clone https://github.com/Chuyi111/mahjong-perception.git
cd mahjong-perception
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\main.py

test