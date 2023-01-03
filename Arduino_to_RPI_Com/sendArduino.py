import RPi.GPIO as GPIO
from lib_nrf24 import NRF24
import time 
import spidev

GPIO.setmode(GPIO.BCM)

pipes = [[0xE8, 0xE8, 0xF0, 0xF0, 0xE1], [ 0xF0, 0xF0, 0xF0, 0xF0, 0xE1]]

radio = NRF24(GPIO, spidev.SpiDev())
radio.begin(0,4)

radio.setPayloadSize(32)
radio.setChannel(0x76)
radio.setDataRate(NRF24.BR_1MBPS)
radio.setPALevel(NRF24.PA_MIN)

radio.setAutoAck(True)
radio.enableDynamicPayloads()
radio.enableAckPayload()

radio.openReadingPipe(1,pipes[1])
radio.printDetails()
radio.startListening()

while True:
	message = list("Hello World")
	radio.write(message)
	print("We sent the message of {}".format(message))

	if radio.isAckPayloadAvailable():
		returnedPL = []
		radio.read(returnedPL, radio.getDynamicPayloadSize())
		print("Our returned payload was {}".format(returnedPL))
	else:
		print("No payload received")
	time.sleep(1)
