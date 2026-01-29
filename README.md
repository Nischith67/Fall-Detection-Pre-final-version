# Elderly Fall Detection and Activity Monitoring System

A real-time system for monitoring elderly activity and detecting falls using YOLO11-pose, Person Re-Identification (ReID), and Face Recognition.

## Features
- **Real-time Activity Classification**: Detects Walking, Standing, Sitting, and Sleeping.
- **Advanced Fall Detection**: Differentiates between Minor and Major falls with escalation logic.
- **Identity Persistence**: Maintains person identity across sessions using ReID and manual naming.
- **Web Dashboard**: Modern, responsive UI for monitoring current status and historical data.
- **Low Power Mode**: Automatically reduces resource usage when no motion is detected.

## Requirements
- Python 3.8+
- Webcam or Video Stream
- NVIDIA GPU (Optional, but recommended for real-time performance)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```bash
   python smart_fall_activity_report.py
   ```

2. Access the dashboard:
   Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Registering Names:
   - You can name detected persons through the web interface.
   - The system will attempt to remember them using body features (ReID).

## Project Structure
- `smart_fall_activity_report.py`: Main application logic.
- `requirements.txt`: Python dependencies.
- `yolo11n-pose.pt`: YOLO11 pose estimation model.
- `monitor_data.db`: SQLite database for activity and fall history (automatically created).

## Configuration
- Adjust detection thresholds in `smart_fall_activity_report.py` if needed.
- Modify `FALL_URL` to point to external notification services.
