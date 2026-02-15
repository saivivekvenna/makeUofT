/*
 * Arduino Sketch: Sensor Reader + Servo Compass
 *
 * Sends sensor data over serial as CSV:
 *   heart_rate, left, right, behind, speak_button
 *
 * Listens for servo commands from Pi5:
 *   SERVO:<angle>\n      (angle 0-180)
 *
 * Wiring:
 *   - Pulse sensor       -> A0
 *   - Left IR/ultrasonic  -> digital pin 2
 *   - Right IR/ultrasonic -> digital pin 3
 *   - Behind IR/ultrasonic-> digital pin 4
 *   - Speak button        -> digital pin 5 (INPUT_PULLUP, active LOW)
 *   - Servo signal        -> digital pin 9 (PWM)
 *
 * Adjust pin numbers and sensor reading logic to match your
 * actual hardware setup.
 */

#include <Servo.h>

// ----- Pin definitions (adjust to match your wiring) -----
#define PULSE_PIN A0
#define LEFT_PIN 2
#define RIGHT_PIN 3
#define BEHIND_PIN 4
#define SPEAK_BTN_PIN 5
#define SERVO_PIN 9

// ----- Timing -----
#define SENSOR_SEND_INTERVAL_MS 100 // send sensor CSV every 100 ms
#define SERIAL_BAUD 9600

// ----- Objects -----
Servo compassServo;

// ----- State -----
unsigned long lastSensorSend = 0;
int currentServoAngle = 90; // start centered (pointing backward toward user)
String serialBuffer = "";

// ----- Heart rate simple moving average -----
#define HR_SAMPLES 10
int hrReadings[HR_SAMPLES];
int hrIndex = 0;
long hrSum = 0;
bool hrReady = false;

void setup()
{
    Serial.begin(SERIAL_BAUD);

    // Sensor inputs
    pinMode(LEFT_PIN, INPUT);
    pinMode(RIGHT_PIN, INPUT);
    pinMode(BEHIND_PIN, INPUT);
    pinMode(SPEAK_BTN_PIN, INPUT_PULLUP); // active LOW

    // Servo
    compassServo.attach(SERVO_PIN);
    compassServo.write(currentServoAngle);

    // Initialize HR buffer
    for (int i = 0; i < HR_SAMPLES; i++)
    {
        hrReadings[i] = 0;
    }

    serialBuffer.reserve(64);
}

void loop()
{
    // --- Read incoming serial commands from Pi5 ---
    while (Serial.available() > 0)
    {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r')
        {
            if (serialBuffer.length() > 0)
            {
                handleCommand(serialBuffer);
                serialBuffer = "";
            }
        }
        else
        {
            if (serialBuffer.length() < 60)
            {
                serialBuffer += c;
            }
        }
    }

    // --- Send sensor data at fixed interval ---
    unsigned long now = millis();
    if (now - lastSensorSend >= SENSOR_SEND_INTERVAL_MS)
    {
        lastSensorSend = now;
        sendSensorData();
    }
}

void handleCommand(const String &cmd)
{
    // Expected format: "SERVO:90"
    if (cmd.startsWith("SERVO:"))
    {
        int angle = cmd.substring(6).toInt();
        angle = constrain(angle, 0, 180);
        if (angle != currentServoAngle)
        {
            currentServoAngle = angle;
            compassServo.write(currentServoAngle);
        }
    }
    // Add more commands here as needed (e.g., "RESET", "LED:on")
}

void sendSensorData()
{
    // --- Heart rate (analog pulse sensor) ---
    int rawPulse = analogRead(PULSE_PIN);
    // Simple BPM estimation: map raw ADC to approximate BPM range.
    // Replace this with proper peak-detection if using a real pulse sensor library.
    int approxBPM = map(rawPulse, 0, 1023, 50, 120);

    // Smooth with moving average
    hrSum -= hrReadings[hrIndex];
    hrReadings[hrIndex] = approxBPM;
    hrSum += approxBPM;
    hrIndex = (hrIndex + 1) % HR_SAMPLES;
    if (hrIndex == 0)
        hrReady = true;
    int heartRate = hrReady ? (int)(hrSum / HR_SAMPLES) : approxBPM;

    // --- Directional sensors ---
    int leftVal = digitalRead(LEFT_PIN);
    int rightVal = digitalRead(RIGHT_PIN);
    int behindVal = digitalRead(BEHIND_PIN);

    // --- Speak button (active LOW) ---
    int speakBtn = (digitalRead(SPEAK_BTN_PIN) == LOW) ? 1 : 0;

    // --- Send CSV line ---
    Serial.print(heartRate);
    Serial.print(",");
    Serial.print(leftVal);
    Serial.print(",");
    Serial.print(rightVal);
    Serial.print(",");
    Serial.print(behindVal);
    Serial.print(",");
    Serial.println(speakBtn);
}
