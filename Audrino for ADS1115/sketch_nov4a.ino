#include <Adafruit_ADS1X15.h>

#include <Wire.h>

Adafruit_ADS1015 ads1015;

void setup(void)
{
  Serial.begin(9600);
  Serial.print("Hello!\n");
  
  Serial.print("Getting differential reading from AIN0 (P) and AIN1 (N)\n");
  Serial.print("ADC Range: +/- 6.144V (1 bit = 3mV)\n");
  ads1015.begin();
  
  ads1015.setGain(GAIN_SIXTEEN);
}

void loop(void)
{
  int16_t results;

  results = ads1015.readADC_Differential_0_1();
  //Serial.print("Differential: ");
  Serial.println(results);
  //Serial.print("("); Serial.print(results * 3); Serial.println("mV)");

  delay(100);
}