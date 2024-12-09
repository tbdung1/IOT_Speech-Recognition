#include <SoftwareSerial.h>
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // Kết nối serial
  pinMode(8, OUTPUT); // Chân 8 nối với loa hoặc buzzer

}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); // Đọc ký tự đầu tiên từ Python
    if (command == '1') {
      digitalWrite(8, HIGH); 
      delay(500);
      digitalWrite(8, LOW); 
    }
  }
}
