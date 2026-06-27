/*
 * HCSR-04 displacement streamer for the ML pipeline.
 * Streams one displacement value (mm) per line over USB serial.
 * Powered entirely by USB 5V -- does NOT need the 7.26V electromagnet supply.
 *
 * Wiring:
 *   HC-SR04 VCC -> 5V
 *   HC-SR04 GND -> GND
 *   HC-SR04 TRIG -> D9
 *   HC-SR04 ECHO -> D10
 */

const int TRIG = 9;
const int ECHO = 10;

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
}

float readDistanceMM() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long duration = pulseIn(ECHO, HIGH, 30000UL); // 30 ms timeout
  if (duration == 0) return -1.0;               // no echo
  // speed of sound ~0.343 mm/us, round trip -> divide by 2
  return (duration * 0.343) / 2.0;
}

void loop() {
  float mm = readDistanceMM();
  if (mm > 0) {
    Serial.println(mm, 1);
  }
  delay(20); // ~50 Hz, matches FS in the Python pipeline
}
