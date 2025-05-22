# 🎨 Color Detection & LED Control System

A real-time color detection system that uses computer vision to detect red, yellow, and green colors from a camera feed and automatically controls corresponding LEDs connected to an Arduino board.

## 🌟 Features

- **Real-time Color Detection**: Uses OpenCV and HSV color space for accurate color detection
- **Arduino LED Control**: Automatically controls RGB LEDs based on detected colors
- **Multiple Camera Sources**: Supports both IP cameras (Android phones) and local webcams
- **Web Interface**: Modern, responsive web interface with glassmorphism design
- **Manual Control**: Override automatic detection with manual LED control
- **Live Video Streaming**: Real-time video feed displayed on web interface
- **Status Monitoring**: Live status updates of detection state and current color

## 🛠️ Hardware Requirements

### Arduino Components:
- Arduino Uno/Nano/ESP32 (any compatible board)
- 3x LEDs (Red, Yellow, Green)
- 3x 220Ω Resistors
- Breadboard
- Jumper wires
- USB cable for Arduino connection

### Camera Options:
- **Option 1**: Android phone with IP Webcam app
- **Option 2**: USB webcam connected to computer

## 📋 Software Requirements

### Python Dependencies:
```bash
pip install opencv-python
pip install numpy
pip install pyserial
pip install flask
pip install requests
```

### Android App (for IP Camera):
- Install "IP Webcam" app from Google Play Store

## 🔧 Hardware Setup

### Arduino Wiring:
```
LED Connections:
- Yellow LED → Pin 9 (with 220Ω resistor)
- Red LED    → Pin 10 (with 220Ω resistor)  
- Green LED  → Pin 11 (with 220Ω resistor)
- All LED cathodes → GND
```

### Wiring Diagram:
```
Arduino Uno    LED + Resistor
Pin 9  -------- Yellow LED ----[220Ω]---- GND
Pin 10 -------- Red LED    ----[220Ω]---- GND  
Pin 11 -------- Green LED  ----[220Ω]---- GND
```

## 📱 Camera Setup

### Option 1: IP Camera (Android Phone)
1. Install "IP Webcam" app on Android phone
2. Connect phone to same WiFi network as computer
3. Open IP Webcam app and start server
4. Note the IP address shown (e.g., 192.168.1.100:8080)
5. Update `IP_CAMERA_URL` in Python code:
   ```python
   IP_CAMERA_URL = "http://YOUR_PHONE_IP:8080/video"
   ```

### Option 2: USB Webcam
- Simply connect USB webcam to computer
- System will automatically detect and use it as fallback

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/color-detection-led-control.git
cd color-detection-led-control
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Arduino Setup
1. Upload the provided Arduino code to your board
2. Note the COM port (Windows) or device path (Linux/Mac)
3. Update the port in Python code:
   ```python
   arduino = serial.Serial('COM3', 9600, timeout=1)  # Change COM3 to your port
   ```

### 4. Configure Camera
- For IP Camera: Update `IP_CAMERA_URL` with your phone's IP
- For USB Camera: No configuration needed

### 5. Run the System
```bash
python app.py
```

### 6. Access Web Interface
- Local: http://localhost:5000
- Network: http://your-computer-ip:5000

## 📁 Project Structure

```
color-detection-led-control/
│
├── app.py                 # Main Python application
├── arduino_code.ino       # Arduino LED control code
├── templates/
│   └── index.html         # Web interface template
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── images/               # Screenshots and diagrams
    ├── setup_diagram.png
    ├── web_interface.png
    └── hardware_setup.jpg
```

## 🎮 Usage Instructions

### Starting the System:
1. **Connect Hardware**: Ensure Arduino is connected and LEDs are wired correctly
2. **Setup Camera**: Configure IP camera or connect USB webcam
3. **Run Application**: Execute `python app.py`
4. **Open Browser**: Navigate to http://localhost:5000

### Web Interface Controls:
- **▶️ Start Detection**: Begin automatic color detection
- **⏹️ Stop Detection**: Pause automatic detection
- **🎮 Manual Control**: Override with manual LED control
  - 🔴 Red LED
  - 🟡 Yellow LED  
  - 🟢 Green LED
  - ⚫ Turn Off All

### Color Detection:
- Hold colored objects in front of camera
- System detects: Red, Yellow, Green
- Corresponding LEDs light up automatically
- Real-time feedback in web interface

## 🎨 Color Detection Parameters

The system uses HSV color space for detection:

### Red Detection:
```python
Range 1: H(0-10), S(50-255), V(50-255)
Range 2: H(170-180), S(50-255), V(50-255)
```

### Yellow Detection:
```python
Range: H(20-30), S(50-255), V(50-255)
```

### Green Detection:
```python
Range: H(40-80), S(50-255), V(50-255)
```

### Minimum Pixel Threshold:
- 500 pixels required for color detection
- Helps eliminate noise and false positives

## 🔧 Configuration

### Arduino Port Configuration:
```python
# Windows
arduino = serial.Serial('COM3', 9600, timeout=1)

# Linux/Mac  
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
```

### IP Camera Configuration:
```python
IP_CAMERA_URL = "http://192.168.1.100:8080/video"  # Replace with your IP
```

### Camera Settings:
```python
# Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame rate
cap.set(cv2.CAP_PROP_FPS, 30)
```

## 🐛 Troubleshooting

### Arduino Issues:
- **Not connecting**: Check COM port, try different ports
- **LEDs not working**: Verify wiring, check resistor values
- **No response**: Ensure Arduino code is uploaded correctly

### Camera Issues:
- **IP Camera not connecting**: 
  - Verify phone and computer on same WiFi
  - Check IP address is correct
  - Test URL in browser first
- **USB Camera not detected**: 
  - Try different USB ports
  - Close other camera applications
  - Check camera permissions

### Detection Issues:
- **Colors not detected**: 
  - Improve lighting conditions
  - Adjust HSV ranges in code
  - Check minimum pixel threshold
- **False detections**: 
  - Increase pixel threshold
  - Improve color object contrast

### Web Interface Issues:
- **Page not loading**: Check if Flask server is running
- **Video not streaming**: Verify camera initialization
- **Controls not working**: Check browser console for errors

## 🔄 System Flow

```
1. Camera captures frame
   ↓
2. Convert BGR to HSV color space
   ↓  
3. Apply color masks for Red/Yellow/Green
   ↓
4. Count pixels for each color
   ↓
5. Determine dominant color (if > threshold)
   ↓
6. Send command to Arduino
   ↓
7. Arduino controls corresponding LED
   ↓
8. Update web interface status
```

## 📊 Performance

- **Frame Rate**: ~30 FPS (adjustable)
- **Detection Latency**: <100ms
- **Arduino Response**: <50ms
- **Web Interface Update**: 1 second intervals

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV community for computer vision libraries
- Flask framework for web interface
- Arduino community for microcontroller support
- Color detection algorithms and HSV color space concepts

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an [Issue](https://github.com/yourusername/color-detection-led-control/issues)
3. Contact: your.email@example.com

## 🔮 Future Enhancements

- [ ] Support for more colors
- [ ] Mobile app for remote control
- [ ] Machine learning based color detection
- [ ] Multiple Arduino board support
- [ ] Data logging and analytics
- [ ] Voice control integration
- [ ] Gesture recognition

---

⭐ **pengolahan citra kelompok 2 6ti5** ⭐
