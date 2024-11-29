#include <SoftwareSerial.h>
#include <MPU6050.h>

SoftwareSerial BTSerial(10,11); // RX | TX
MPU6050 mpu;
int ledPin = 9;
char receivedChar;
const int flexSensorPins[] = {A0, A1, A2, A3,A6}; // Flex传感器引脚

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  BTSerial.begin(9600);
}

void loop() {
  if (BTSerial.available()) {
    receivedChar = BTSerial.read();

    if (receivedChar == '1') {
      digitalWrite(ledPin, HIGH);
      BTSerial.println("LED 点亮");
    } else if (receivedChar == '0') {
      digitalWrite(ledPin, LOW);
      BTSerial.println("LED 熄灭");
    }
  // 加速度计和陀螺仪数据
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  String data = String(ax) + String(ay) + String(az) + String(gx) + String(gy) + String(gz);
  BTSerial.print(data);

  // 弯曲传感器数据
  int A0_raw, A1_raw, A2_raw, A3_raw, A6_raw;
  A0_raw = analogRead(flexSensorPins[0]);
  A1_raw = analogRead(flexSensorPins[1]);
  A2_raw = analogRead(flexSensorPins[2]);
  A3_raw = analogRead(flexSensorPins[3]);
  A6_raw = analogRead(flexSensorPins[4]);
  
  data = String(A0_raw) + String(A1_raw) + String(A2_raw) + String(A3_raw) + String(A6_raw);
  BTSerial.print(data);

  //
  delay(1000);

  }
}