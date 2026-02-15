/*
 * Arduino Sketch: Sensor Reader + Servo Compass
 *
 * Based on your existing sensor code — only addition is a servo
 * on pin 10 that listens for "SERVO:<angle>\n" commands from the Pi5.
 *
 * Hardware (unchanged):
 *   - HC-SR04 Left      -> trig=8, echo=9
 *   - HC-SR04 Right     -> trig=5, echo=6
 *   - HC-SR04 Center    -> trig=2, echo=3
 *   - Left buzzer       -> pin 11
 *   - Right buzzer      -> pin 13
 *   - Center buzzer     -> pin 12
 *   - Pulse sensor      -> A0
 *   - PTT button        -> pin 7 (INPUT_PULLUP)
 *
 * New hardware (one wire + power):
 *   - Servo signal      -> pin 10 (PWM)
 *   - Servo VCC         -> 5V
 *   - Servo GND         -> GND
 */

#include <HCSR04.h>
#include <Servo.h>

// --- Ultrasonic sensors (trig, echo) ---
HCSR04 hcLeft(8, 9);
HCSR04 hcRight(5, 6);
HCSR04 hcCenter(2, 3);

// --- Pin definitions ---
const int lbuzz = 11, rbuzz = 13, cbuzz = 12;
const int heartPin = A0;
const int PTT = 7;
const int SERVO_PIN = 10; // only free PWM pin

// --- Timing ---
unsigned long previousMillis = 0;
const long interval = 5; // 5 ms = 200 Hz sensor loop

// --- Servo state ---
Servo compassServo;
int currentServoAngle = 90; // centered = pointing backward toward user
String serialBuffer = "";

void setup()
{
    pinMode(lbuzz, OUTPUT);
    pinMode(rbuzz, OUTPUT);
    pinMode(cbuzz, OUTPUT);
    pinMode(PTT, INPUT_PULLUP);

    compassServo.attach(SERVO_PIN);
    compassServo.write(currentServoAngle);

    serialBuffer.reserve(64);
    Serial.begin(115200);
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

    // --- 200 Hz sensor loop (unchanged timing) ---
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= interval)
    {
        previousMillis = currentMillis;

        // 1. Heart rate (raw analog)
        int heartVal = analogRead(heartPin);

        // 2. Ultrasonic distances
        float dLeft = hcLeft.dist();
        float dRight = hcRight.dist();
        float dCenter = hcCenter.dist();

        // 3. Proximity logic + buzzers
        int L = (dLeft > 0.1 && dLeft < 50.0) ? 1 : 0;
        int R = (dRight > 0.1 && dRight < 50.0) ? 1 : 0;
        int C = (dCenter > 0.1 && dCenter < 400.0) ? 1 : 0;

        digitalWrite(lbuzz, L);
        digitalWrite(rbuzz, R);
        digitalWrite(cbuzz, C);

        // 4. PTT button (INPUT_PULLUP → pressed = LOW)
        int pttState = (digitalRead(PTT) == LOW) ? 0 : 1;

        // 5. CSV output: HeartRate,L,R,C,PTT
        Serial.print(heartVal);
        Serial.print(",");
        Serial.print(L);
        Serial.print(",");
        Serial.print(R);
        Serial.print(",");
        Serial.print(C);
        Serial.print(",");
        Serial.println(pttState);
    }
}

void handleCommand(const String &cmd)
{
    // Format from Pi5: "SERVO:90"
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
}
