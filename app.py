import cv2
import numpy as np
import serial
import time
import os
import sys
import json
import requests
import warnings
import signal

from flask import Flask, render_template, Response, jsonify
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from skimage import feature, filters, measure
from datetime import datetime
from flask import request
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
captured_images_data = []

# IP Camera configuration - SESUAIKAN DENGAN IP ANDROID ANDA
IP_CAMERA_URL = "http://192.168.8.130:8080/video"  # Ganti dengan IP yang benar

# Pastikan folder captures ada
CAPTURES_DIR = 'static/captures'
if not os.path.exists(CAPTURES_DIR):
    os.makedirs(CAPTURES_DIR)

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

@app.route('/test_camera')
def test_camera():
    """Test endpoint untuk mengecek status kamera"""
    global cap, frame
    
    try:
        if cap is None or not cap.isOpened():
            return jsonify({
                'success': False,
                'error': 'Kamera tidak aktif',
                'camera_status': 'disconnected'
            })
        
        # Test baca frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            return jsonify({
                'success': False,
                'error': 'Tidak dapat membaca frame dari kamera',
                'camera_status': 'error'
            })
        
        return jsonify({
            'success': True,
            'camera_status': 'connected',
            'frame_size': f"{test_frame.shape[1]}x{test_frame.shape[0]}",
            'frame_available': frame is not None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'camera_status': 'error'
        })
    
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

@app.route('/capture_image')
def capture_image():
    """Endpoint untuk mengambil foto dari video stream"""
    try:
        # Ambil parameter dari request
        format_type = request.args.get('format', 'jpg').lower()
        quality = request.args.get('quality', 'medium').lower()
        
        # PERBAIKAN 1: Gunakan variabel global 'cap' yang sudah didefinisikan
        global cap, frame
        
        # PERBAIKAN 2: Cek apakah kamera sudah aktif
        if cap is None or not cap.isOpened():
            return jsonify({
                'success': False,
                'error': 'Kamera tidak aktif'
            })
        
        # PERBAIKAN 3: Gunakan frame global yang sudah ada atau ambil frame baru
        current_frame = None
        if frame is not None:
            current_frame = frame.copy()
        else:
            # Fallback: ambil frame langsung dari kamera
            ret, current_frame = cap.read()
            if not ret or current_frame is None:
                return jsonify({
                    'success': False,
                    'error': 'Gagal mengambil frame dari kamera'
                })
        
        # Generate filename dengan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.{format_type}"
        filepath = os.path.join(CAPTURES_DIR, filename)
        
        # Set kualitas berdasarkan parameter
        if format_type == 'jpg':
            if quality == 'high':
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif quality == 'low':
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            else:  # medium
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        else:  # png
            if quality == 'high':
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            elif quality == 'low':
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:  # medium
                success = cv2.imwrite(filepath, current_frame, [cv2.IMWRITE_PNG_COMPRESSION, 5])
        
        # PERBAIKAN 4: Cek apakah penulisan file berhasil
        if not success:
            return jsonify({
                'success': False,
                'error': 'Gagal menyimpan file gambar'
            })
        
        # PERBAIKAN 5: Cek apakah file benar-benar tersimpan
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File gambar tidak tersimpan'
            })
        
        # Hitung ukuran file
        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Deteksi warna pada area tengah frame
        detected_color = detect_color_in_frame(current_frame)
        
        # Simpan informasi gambar
        image_info = {
            'id': len(captured_images_data) + 1,
            'filename': filename,
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'color': detected_color,
            'format': format_type.upper(),
            'size': size_str,
            'file_size': file_size
        }
        
        # Tambahkan ke list (maksimal 50 gambar)
        captured_images_data.insert(0, image_info)
        if len(captured_images_data) > 50:
            # Hapus gambar lama dari disk
            old_image = captured_images_data.pop()
            old_filepath = os.path.join(CAPTURES_DIR, old_image['filename'])
            if os.path.exists(old_filepath):
                try:
                    os.remove(old_filepath)
                except Exception as e:
                    print(f"Warning: Could not remove old file {old_filepath}: {e}")
        
        print(f"✓ Photo captured successfully: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'detected_color': detected_color,
            'file_size': size_str,
            'timestamp': image_info['timestamp']
        })
        
    except Exception as e:
        print(f"Error in capture_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def detect_color_in_frame(frame):
    """Deteksi warna pada area tengah frame"""
    try:
        h, w = frame.shape[:2]
        
        # Ambil area tengah (150x150 pixel)
        center_x, center_y = w // 2, h // 2
        roi_size = 75
        
        # PERBAIKAN: Pastikan ROI tidak keluar dari batas frame
        roi_x1 = max(0, center_x - roi_size)
        roi_y1 = max(0, center_y - roi_size)
        roi_x2 = min(w, center_x + roi_size)
        roi_y2 = min(h, center_y + roi_size)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return "Tidak terdeteksi"
        
        # Convert ke HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Definisi range warna dalam HSV
        color_ranges = {
            'Merah': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([160, 50, 50]), np.array([180, 255, 255]))
            ],
            'Kuning': [(np.array([20, 50, 50]), np.array([30, 255, 255]))],
            'Hijau': [(np.array([40, 50, 50]), np.array([80, 255, 255]))]
        }
        
        max_pixels = 0
        detected_color = "Tidak terdeteksi"
        
        for color_name, ranges in color_ranges.items():
            total_pixels = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                total_pixels += cv2.countNonZero(mask)
            
            if total_pixels > max_pixels and total_pixels > 500:  # threshold
                max_pixels = total_pixels
                detected_color = color_name
        
        return detected_color
        
    except Exception as e:
        print(f"Error in detect_color_in_frame: {str(e)}")
        return "Error"

@app.route('/clear_images', methods=['POST'])
def clear_images():
    """Endpoint untuk menghapus semua foto"""
    try:
        global captured_images_data
        
        # Hapus semua file dari disk
        deleted_count = 0
        for image_info in captured_images_data:
            filepath = os.path.join(CAPTURES_DIR, image_info['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted_count += 1
        
        # Kosongkan list
        captured_images_data.clear()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Berhasil menghapus {deleted_count} foto'
        })
        
    except Exception as e:
        print(f"Error in clear_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_captured_images')
def get_captured_images():
    """Endpoint untuk mendapatkan daftar foto yang sudah diambil"""
    try:
        return jsonify({
            'success': True,
            'images': captured_images_data,
            'total': len(captured_images_data)
        })
        
    except Exception as e:
        print(f"Error in get_captured_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'images': [],
            'total': 0
        })

# Tambahkan juga route untuk serving static files jika belum ada
@app.route('/static/captures/<filename>')
def serve_capture(filename):
    """Serve captured images"""
    try:
        from flask import send_from_directory
        return send_from_directory(CAPTURES_DIR, filename)
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return "File not found", 404
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['GET', 'POST'])
def start_detection():
    global is_detecting
    is_detecting = True
    print("🎯 Detection started")
    return jsonify({"status": "Detection started", "detecting": True})

@app.route('/stop_detection', methods=['GET', 'POST'])
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

@app.route('/manual_control/<color>', methods=['GET', 'POST'])
def manual_control(color):
    valid_colors = ['MERAH', 'KUNING', 'HIJAU', 'OFF']
    if color.upper() in valid_colors:
        send_to_arduino(color.upper())
        print(f"📱 Manual control: {color}")
        return jsonify({"status": f"LED {color} controlled manually"})
    return jsonify({"error": "Invalid color"}), 400
@app.route('/control_led/<action>', methods=['GET', 'POST'])
def control_led(action):
    """Route untuk kontrol LED manual"""
    led_commands = {
        'red': 'MERAH',
        'yellow': 'KUNING', 
        'green': 'HIJAU',
        'off': 'OFF',
        'merah': 'MERAH',
        'kuning': 'KUNING',
        'hijau': 'HIJAU'
    }
    
    action_lower = action.lower()
    if action_lower in led_commands:
        command = led_commands[action_lower]
        send_to_arduino(command)
        print(f"🔧 Manual LED Control: {action} -> {command}")
        return jsonify({"status": f"LED {action} controlled manually"})
    else:
        return jsonify({"error": "Invalid LED action"}), 400

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

@app.route('/analyze_image/histogram/<filename>')
def analyze_histogram(filename):
    try:
        # Baca gambar
        image_path = os.path.join('static/captures', filename)
        image = cv2.imread(image_path)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Image not found'})
        
        # Hitung histogram
        color = ('b', 'g', 'r')
        plt.figure(figsize=(10, 6))
        
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=f'{col.upper()} Channel')
        
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'RGB Histogram - {filename}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1e3c72', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        histogram_html = f'<img src="data:image/png;base64,{plot_data}" style="width: 100%;">'
        
        return jsonify({
            'success': True,
            'histogram_html': histogram_html
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze_image/colorspace/<filename>')
def analyze_colorspace(filename):
    try:
        image_path = os.path.join('static/captures', filename)
        image = cv2.imread(image_path)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Image not found'})
        
        # Konversi ke berbagai color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Buat subplot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('#1e3c72')
        
        # Original BGR
        axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original (RGB)', color='white')
        axes[0,0].axis('off')
        
        # HSV channels
        for i, (channel, title) in enumerate(zip(cv2.split(hsv), ['Hue', 'Saturation', 'Value'])):
            axes[0,i+1].imshow(channel, cmap='gray')
            axes[0,i+1].set_title(f'HSV - {title}', color='white')
            axes[0,i+1].axis('off')
        
        # LAB channels
        for i, (channel, title) in enumerate(zip(cv2.split(lab), ['L*', 'a*', 'b*'])):
            axes[1,i].imshow(channel, cmap='gray')
            axes[1,i].set_title(f'LAB - {title}', color='white')
            axes[1,i].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1e3c72', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        colorspace_html = f'<img src="data:image/png;base64,{plot_data}" style="width: 100%;">'
        
        return jsonify({
            'success': True,
            'colorspace_html': colorspace_html
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze_image/edges/<filename>')
def analyze_edges(filename):
    try:
        image_path = os.path.join('static/captures', filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Image not found'})
        
        # Berbagai metode edge detection
        canny = cv2.Canny(image, 50, 150)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Buat subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#1e3c72')
        
        axes[0,0].imshow(image, cmap='gray')
        axes[0,0].set_title('Original', color='white')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(canny, cmap='gray')
        axes[0,1].set_title('Canny Edge Detection', color='white')
        axes[0,1].axis('off')
        
        axes[1,0].imshow(sobel_x, cmap='gray')
        axes[1,0].set_title('Sobel X', color='white')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(sobel_combined, cmap='gray')
        axes[1,1].set_title('Sobel Combined', color='white')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1e3c72', edgecolor='none')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        edges_html = f'<img src="data:image/png;base64,{plot_data}" style="width: 100%;">'
        
        return jsonify({
            'success': True,
            'edges_html': edges_html
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze_image/features/<filename>')
def analyze_features(filename):
    try:
        image_path = os.path.join('static/captures', filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Image not found'})
        
        # Hitung berbagai features
        mean_rgb = np.mean(image, axis=(0,1))
        std_rgb = np.std(image, axis=(0,1))
        
        # Texture features menggunakan GLCM (simplified)
        contrast = np.var(gray)
        brightness = np.mean(gray)
        
        # Shape features (contours)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features_html = f"""
        <div style="color: white; font-family: Arial, sans-serif;">
            <h4 style="color: #4ecdc4; margin-bottom: 20px;">📊 Image Features Analysis</h4>
            
            <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="color: #ffd93d;">🎨 Color Features</h5>
                <p><strong>Mean RGB:</strong> R:{mean_rgb[2]:.2f}, G:{mean_rgb[1]:.2f}, B:{mean_rgb[0]:.2f}</p>
                <p><strong>Std RGB:</strong> R:{std_rgb[2]:.2f}, G:{std_rgb[1]:.2f}, B:{std_rgb[0]:.2f}</p>
            </div>
            
            <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="color: #51cf66;">🔍 Texture Features</h5>
                <p><strong>Brightness:</strong> {brightness:.2f}</p>
                <p><strong>Contrast:</strong> {contrast:.2f}</p>
            </div>
            
            <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h5 style="color: #ff6b6b;">📐 Shape Features</h5>
                <p><strong>Number of Contours:</strong> {len(contours)}</p>
                <p><strong>Image Size:</strong> {gray.shape[1]} x {gray.shape[0]} pixels</p>
                <p><strong>Total Pixels:</strong> {gray.shape[0] * gray.shape[1]:,}</p>
            </div>
        </div>
        """
        
        return jsonify({
            'success': True,
            'features_html': features_html
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

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
            print(f"Network access: http://192.168.1.118:5000")
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