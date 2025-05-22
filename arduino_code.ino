// Arduino Code untuk mengontrol LED berdasarkan deteksi warna
// Pin definitions
const int LED_KUNING = 9;
const int LED_MERAH = 10;
const int LED_HIJAU = 11;

void setup()
{
    // Initialize serial communication
    Serial.begin(9600);

    // Set LED pins as outputs
    pinMode(LED_KUNING, OUTPUT);
    pinMode(LED_MERAH, OUTPUT);
    pinMode(LED_HIJAU, OUTPUT);

    // Turn off all LEDs initially
    digitalWrite(LED_KUNING, LOW);
    digitalWrite(LED_MERAH, LOW);
    digitalWrite(LED_HIJAU, LOW);

    Serial.println("Arduino LED Controller Ready");
}

void loop()
{
    // Check if data is available from serial
    if (Serial.available() > 0)
    {
        String command = Serial.readStringUntil('\n');
        command.trim(); // Remove whitespace

        // Turn off all LEDs first
        digitalWrite(LED_KUNING, LOW);
        digitalWrite(LED_MERAH, LOW);
        digitalWrite(LED_HIJAU, LOW);

        // Control LEDs based on received command
        if (command == "KUNING")
        {
            digitalWrite(LED_KUNING, HIGH);
            Serial.println("LED Kuning ON");
        }
        else if (command == "MERAH")
        {
            digitalWrite(LED_MERAH, HIGH);
            Serial.println("LED Merah ON");
        }
        else if (command == "HIJAU")
        {
            digitalWrite(LED_HIJAU, HIGH);
            Serial.println("LED Hijau ON");
        }
        else if (command == "OFF")
        {
            // All LEDs already turned off above
            Serial.println("All LEDs OFF");
        }
        else
        {
            Serial.println("Unknown command: " + command);
        }
    }

    delay(100); // Small delay to prevent overwhelming the serial buffer
}