/*
 * Arduino Sketch: Sensor Reader
 *
 * Sends sensor data over serial as CSV at 200 Hz:
 *   HeartRate, L, R, C, PTT
 *
 * Hardware:
 *   - HC-SR04 Left      -> trig=8, echo=9
 *   - HC-SR04 Right     -> trig=5, echo=6
 *   - HC-SR04 Center    -> trig=2, echo=3
 *   - Left buzzer       -> pin 11
 *   - Right buzzer      -> pin 13
 *   - Center buzzer     -> pin 12
 *   - Pulse sensor      -> A0
 *   - PTT button        -> pin 7 (INPUT_PULLUP)
 *
 * Servo compass is driven directly by the Pi5 via GPIO —
 * no servo code needed here.
 */

#include <HCSR04.h>

// Initialize sensors (trig, echo)
HCSR04 hcLeft(8, 9);
HCSR04 hcRight(5, 6);
HCSR04 hcCenter(2, 3);

// Pin Definitions
const int lbuzz = 11, rbuzz = 13, cbuzz = 12;
const int heartPin = A0;
const int PTT = 7;

// Timing Variables
unsigned long previousMillis = 0;
const long interval = 5; // 5ms = 200Hz exact sampling

void setup()
{
    pinMode(lbuzz, OUTPUT);
    pinMode(rbuzz, OUTPUT);
    pinMode(cbuzz, OUTPUT);
    pinMode(PTT, INPUT_PULLUP);

    Serial.begin(115200);
}

void loop()
{
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= interval)
    {
        previousMillis = currentMillis;

        // 1. Read Heart Rate
        int heartVal = analogRead(heartPin);

        // 2. Read Distances
        float dLeft = hcLeft.dist();
        float dRight = hcRight.dist();
        float dCenter = hcCenter.dist();

        // 3. Logic & Buzzers
        int L = (dLeft > 0.1 && dLeft < 50.0) ? 1 : 0;
        int R = (dRight > 0.1 && dRight < 50.0) ? 1 : 0;
        int C = (dCenter > 0.1 && dCenter < 400.0) ? 1 : 0;

        digitalWrite(lbuzz, L);
        digitalWrite(rbuzz, R);
        digitalWrite(cbuzz, C);

        // 4. PTT Button (INPUT_PULLUP -> pressed = LOW)
        int pttState = (digitalRead(PTT) == LOW) ? 0 : 1;

        // 5. CSV Output: HeartRate,L,R,C,PTT
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
