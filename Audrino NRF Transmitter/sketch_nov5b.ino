/*
* Arduino Wireless Communication Tutorial
*     Example 1 - Transmitter Code
*                
* by Dejan Nedelkovski, www.HowToMechatronics.com
* 
* Library: TMRh20/RF24, https://github.com/tmrh20/RF24/
*/

#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>

#include <Adafruit_ADS1X15.h>
#include <Wire.h>

Adafruit_ADS1015 ads1015;
RF24 radio(9, 10); // CE, CSN

const byte address[6] = "00001";

void setup() {
  radio.begin();
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();

  Serial.begin(9600);
  Serial.print("Hello!\n");
  
  Serial.print("Getting differential reading from AIN0 (P) and AIN1 (N)\n");
  Serial.print("ADC Range: +/- 6.144V (1 bit = 3mV)\n");
  ads1015.begin();
  
  ads1015.setGain(GAIN_SIXTEEN);
}

void loop() {
  int16_t results;
  results = ads1015.readADC_Differential_0_1();
  //Serial.print("Differential: ");
  Serial.println(results);
  //Serial.print("("); Serial.print(results * 3); Serial.println("mV)");
  delay(10);

  const int text[1] = {results};
  radio.write(&text, sizeof(text));
  delay(100);
}