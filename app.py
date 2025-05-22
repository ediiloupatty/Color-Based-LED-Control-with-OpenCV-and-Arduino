import cv2
import numpy as np
import serial
import time
from flask import Flask, render_template, Response, jsonify
import threading
import requests
import warnings
import signal
import sys
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables
arduino = None
current_color = "Tidak ada"
frame = None
is_detecting = False
cap = None
camera_source = "Not connected"
shutdown_flag = False

# IP Camera configuration - SESUAIKAN DENGAN IP ANDROID ANDA
IP_CAMERA_URL = "http://192.168.8.130:8080/video"  # Ganti dengan IP yang benar

# Signal handler untuk graceful shutdown
def signal_handler(sig, frame):
    global shutdown_flag
    print("\n🛑 Menerima sinyal shutdown...")
    shutdown_flag = True
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Initialize Arduino connection
def init_arduino():
    global arduino
    try:
        # Ganti 'COM3' dengan port Arduino Anda (bisa COM4, COM5, dll)
        arduino = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)
        print("✓ Arduino connected successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: Arduino not connected - {e}")
        return False

# Send command to Arduino
def send_to_arduino(command):
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.write((command + '\n').encode())
            response = arduino.readline().decode().strip()
            print(f"Arduino: {response}")
        except Exception as e:
            print(f"Arduino error: {e}")

# Test IP Camera connection - IMPROVED dengan timeout singkat
def test_ip_camera():
    try:
        print(f"🔍 Testing IP Camera: {IP_CAMERA_URL}")
        print("   (timeout 2 detik - tekan Ctrl+C untuk skip)")
        
        # Timeout lebih singkat dan stream=True untuk response cepat
        response = requests.get(IP_CAMERA_URL, timeout=2, stream=True)
        
        # Cek status code saja, tidak perlu download content
        if response.status_code == 200:
            print("✓ IP Camera accessible")
            response.close()  # Tutup koneksi
            return True
        else:
            print(f"⚠ IP Camera return status: {response.status_code}")
            response.close()
            return False
            
    except requests.exceptions.Timeout:
        print("⚠ IP Camera timeout (2s) - mungkin lambat atau tidak ada")
        return False
    except requests.exceptions.ConnectionError:
        print("⚠ IP Camera connection error - cek IP address atau WiFi")
        return False
    except KeyboardInterrupt:
        print("\n⏭ Skip IP Camera test - lanjut ke webcam lokal")
        return False
    except Exception as e:
        print(f"⚠ IP Camera error: {e}")
        return False

# Color detection function
def detect_color(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'red': [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ],
        'yellow': [(np.array([20, 50, 50]), np.array([30, 255, 255]))],
        'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
    }
    
    # Count pixels for each color
    color_counts = {}
    
    # Red (dua range)
    mask_red1 = cv2.inRange(hsv, color_ranges['red'][0][0], color_ranges['red'][0][1])
    mask_red2 = cv2.inRange(hsv, color_ranges['red'][1][0], color_ranges['red'][1][1])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    color_counts['red'] = cv2.countNonZero(mask_red)
    
    # Yellow
    mask_yellow = cv2.inRange(hsv, color_ranges['yellow'][0][0], color_ranges['yellow'][0][1])
    color_counts['yellow'] = cv2.countNonZero(mask_yellow)
    
    # Green
    mask_green = cv2.inRange(hsv, color_ranges['green'][0][0], color_ranges['green'][0][1])
    color_counts['green'] = cv2.countNonZero(mask_green)
    
    # Minimum threshold
    threshold = 500
    
    # Find dominant color
    detected_color = "Tidak ada"
    max_pixels = max(color_counts.values())
    
    if max_pixels > threshold:
        for color, count in color_counts.items():
            if count == max_pixels:
                if color == 'red':
                    detected_color = "Merah"
                elif color == 'yellow':
                    detected_color = "Kuning"
                elif color == 'green':
                    detected_color = "Hijau"
                break
    
    return detected_color, mask_red, mask_yellow, mask_green

# Initialize camera - IMPROVED dengan better error handling
def init_camera():
    global cap, camera_source, shutdown_flag
    
    print("🔧 Initializing camera...")
    
    # 1. Test IP Camera dengan timeout handling
    try:
        if test_ip_camera() and not shutdown_flag:
            print("📱 Connecting to IP Camera...")
            cap = cv2.VideoCapture(IP_CAMERA_URL)
            
            # Set timeout untuk OpenCV
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test frame dengan timeout
            print("   Testing frame capture...")
            start_time = time.time()
            ret, test_frame = cap.read()
            
            if ret and test_frame is not None and not shutdown_flag:
                elapsed = time.time() - start_time
                print(f"✓ IP Camera connected successfully ({elapsed:.1f}s)")
                camera_source = f"IP Camera: {IP_CAMERA_URL}"
                return True
            else:
                print("⚠ Cannot read frame from IP Camera")
                if cap:
                    cap.release()
                    cap = None
    except KeyboardInterrupt:
        print("\n⏭ IP Camera setup interrupted - trying local webcam")
        if cap:
            cap.release()
            cap = None
    except Exception as e:
        print(f"⚠ IP Camera error: {e}")
        if cap:
            cap.release()
            cap = None
    
    # Return false jika shutdown
    if shutdown_flag:
        return False
    
    # 2. Fallback ke webcam lokal
    print("💻 Trying local webcam...")
    
    # Suppress OpenCV warnings
    import os
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    
    for i in range(3):  # Test webcam index 0, 1, 2
        if shutdown_flag:
            return False
            
        try:
            print(f"  Testing webcam index {i}...")
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                # Test read dengan timeout
                start_time = time.time()
                ret, test_frame = cap.read()
                elapsed = time.time() - start_time
                
                if ret and test_frame is not None and elapsed < 3.0:
                    print(f"✓ Local webcam connected (index {i}, {elapsed:.1f}s)")
                    camera_source = f"Local Webcam (index {i})"
                    
                    # Settings untuk webcam lokal
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    return True
                else:
                    print(f"  Webcam {i} too slow or no frame ({elapsed:.1f}s)")
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                    cap = None
        except KeyboardInterrupt:
            print("\n⏭ Webcam test interrupted")
            if cap:
                cap.release()
                cap = None
            return False
        except Exception as e:
            print(f"  Webcam {i} error: {e}")
            if cap:
                cap.release()
                cap = None
    
    print("❌ No camera available")
    return False

# Camera capture function - OPTIMIZED dengan better error handling
def capture_frames():
    global frame, current_color, is_detecting, cap, shutdown_flag
    
    if not cap or not cap.isOpened():
        print("❌ Camera not initialized")
        return
    
    previous_color = ""
    frame_count = 0
    consecutive_failures = 0
    max_failures = 10
    
    print("🎥 Starting frame capture...")
    
    while not shutdown_flag:
        try:
            ret, new_frame = cap.read()
            
            if not ret or new_frame is None:
                consecutive_failures += 1
                print(f"⚠ Frame read failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print("⚠ Too many failures, attempting reconnect...")
                    
                    # Reconnect
                    if cap:
                        cap.release()
                        cap = None
                    
                    time.sleep(2)
                    if not shutdown_flag and init_camera():
                        print("✓ Camera reconnected")
                        consecutive_failures = 0
                        continue
                    else:
                        print("❌ Reconnection failed")
                        break
                
                time.sleep(0.1)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Process frame
            frame = new_frame.copy()
            frame_count += 1
            
            # Resize jika perlu
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Add overlay information
            info_color = (0, 255, 255)  # Yellow text
            cv2.putText(frame, f"Source: {camera_source}", (10, frame.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
            cv2.putText(frame, f"Status: {'DETECTING' if is_detecting else 'STOPPED'}", 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if is_detecting else (0, 0, 255), 1)
            
            # Color detection
            if is_detecting:
                detected_color, _, _, _ = detect_color(frame)
                current_color = detected_color
                
                # Send to Arduino only when color changes
                if detected_color != previous_color:
                    arduino_commands = {
                        "Merah": "MERAH",
                        "Kuning": "KUNING", 
                        "Hijau": "HIJAU",
                        "Tidak ada": "OFF"
                    }
                    
                    command = arduino_commands.get(detected_color, "OFF")
                    send_to_arduino(command)
                    previous_color = detected_color
                    print(f"🎯 Color detected: {detected_color}")
                
                # Display detection
                color_display = (0, 255, 0) if detected_color != "Tidak ada" else (0, 0, 255)
                cv2.putText(frame, f"Warna: {detected_color}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_display, 2)
            
            time.sleep(0.033)  # ~30 FPS
            
        except KeyboardInterrupt:
            print("🛑 Stopping capture...")
            break
        except Exception as e:
            print(f"❌ Capture error: {e}")
            time.sleep(1)

# Generate frames for web streaming
def generate_frames():
    global frame, shutdown_flag
    
    no_frame_count = 0
    
    while not shutdown_flag:
        try:
            if frame is not None:
                no_frame_count = 0
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                no_frame_count += 1
                
                # Dummy frame when no camera
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, "No Camera Feed Available", (150, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(dummy_frame, "Check camera connection", (180, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(dummy_frame, f"Attempts: {no_frame_count}", (270, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                ret, buffer = cv2.imencode('.jpg', dummy_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # 30 FPS
            
        except Exception as e:
            print(f"❌ Stream error: {e}")
            time.sleep(1)

# Cleanup function
def cleanup():
    global cap, arduino
    print("\n🧹 Cleaning up...")
    
    if cap:
        cap.release()
        print("✓ Camera released")
    
    if arduino and arduino.is_open:
        send_to_arduino("OFF")  # Turn off LEDs
        arduino.close()
        print("✓ Arduino disconnected")
    
    cv2.destroyAllWindows()
    print("✓ Cleanup completed")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global is_detecting
    is_detecting = True
    print("🎯 Detection started")
    return jsonify({"status": "Detection started", "detecting": True})

@app.route('/stop_detection')
def stop_detection():
    global is_detecting
    is_detecting = False
    send_to_arduino("OFF")
    print("⏹ Detection stopped")
    return jsonify({"status": "Detection stopped", "detecting": False})

@app.route('/get_status')
def get_status():
    camera_status = "Connected" if cap and cap.isOpened() else "Disconnected"
    return jsonify({
        "current_color": current_color,
        "is_detecting": is_detecting,
        "camera_status": camera_status,
        "camera_source": camera_source
    })

@app.route('/manual_control/<color>')
def manual_control(color):
    valid_colors = ['MERAH', 'KUNING', 'HIJAU', 'OFF']
    if color.upper() in valid_colors:
        send_to_arduino(color.upper())
        print(f"📱 Manual control: {color}")
        return jsonify({"status": f"LED {color} controlled manually"})
    return jsonify({"error": "Invalid color"}), 400

@app.route('/reconnect_camera')
def reconnect_camera():
    global cap
    try:
        print("🔄 Reconnecting camera...")
        if cap:
            cap.release()
            cap = None
        
        time.sleep(1)
        
        if init_camera():
            return jsonify({"status": "Camera reconnected successfully", 
                          "source": camera_source})
        else:
            return jsonify({"error": "Failed to reconnect camera"}), 500
    except Exception as e:
        return jsonify({"error": f"Reconnection error: {str(e)}"}), 500

# Main execution
if __name__ == '__main__':
    try:
        print("=" * 50)
        print("🎨 COLOR DETECTION SYSTEM")
        print("=" * 50)
        
        # System check
        print("🔧 System Check:")
        
        # 1. Arduino
        if init_arduino():
            arduino_status = "✓ Connected"
        else:
            arduino_status = "⚠ Not connected (LED control disabled)"
        print(f"   Arduino: {arduino_status}")
        
        # 2. Camera
        print("   Camera: Initializing...")
        print("   💡 Tips: Tekan Ctrl+C kapan saja untuk skip ke step berikutnya")
        
        if init_camera() and not shutdown_flag:
            print(f"   Camera: ✓ {camera_source}")
            
            # Start camera thread
            print("🎥 Starting camera thread...")
            camera_thread = threading.Thread(target=capture_frames, daemon=True)
            camera_thread.start()
            
            # Server info
            print("\n" + "=" * 50)
            print("🌐 WEB SERVER INFORMATION")
            print("=" * 50)
            print(f"Local access: http://localhost:5000")
            print(f"Network access: http://192.168.8.112:5000")
            print(f"IP Camera URL: {IP_CAMERA_URL}")
            print("\n📱 IP Camera Setup:")
            print("   1. Install 'IP Webcam' app on Android")
            print("   2. Connect to same WiFi network")
            print("   3. Start server in app")
            print("   4. Use IP address shown in app")
            print("\n⌨️  Controls:")
            print("   - Start/Stop Detection via web interface")
            print("   - Manual LED control available")
            print("   - Real-time color detection")
            print(f"\n🛑 Press Ctrl+C to stop server")
            print("=" * 50)
            
            # Start Flask
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
            
        else:
            if not shutdown_flag:
                print("   Camera: ❌ Failed to initialize")
                print("\n🔧 TROUBLESHOOTING:")
                print("   IP Camera:")
                print("   - Check Android IP Webcam app is running")
                print("   - Verify IP address is correct") 
                print("   - Test URL in browser first")
                print("   - Ensure same WiFi network")
                print("   - Try manual skip with Ctrl+C")
                print("\n   Local Webcam:")
                print("   - Check if webcam is connected")
                print("   - Close other camera applications")
                print("   - Try different USB ports")
                print("\n💡 Current settings:")
                print(f"   IP Camera URL: {IP_CAMERA_URL}")
                print("   Arduino Port: COM3")
            else:
                print("   Camera: ⏭ Initialization cancelled by user")
        
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        # Cleanup
        shutdown_flag = True
        cleanup()