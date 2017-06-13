import loadcell as lc #import load cell library
import RPi.GPIO as GPIO
import lcd	#import lcd library
import kpad	#import keypad library
import time
import os
import math
import datetime
import sys
import json
import urllib.request
import requests

#NETWORK Constants
URL = "http://192.168.0.66:8000/machine/"
PREDICT_URL = "http://192.168.0.66:8000/predict/"
USER_ID = 1
PASSWORD = "random"
imagepath = "/home/pi/ghfarm/images/"


#address constant for lines in lcd display
LINE_1 = 0x80
LINE_2 = 0xC0
LINE_3 = 0x94
LINE_4 = 0xD4

LINES = [LINE_1,LINE_2,LINE_3,LINE_4]

#Global Variables
min_weight = 10
baseValue = 0	#variable to store the base value of load cell
taredWeight = 0	#variable to store tared weight
DOUT = 22	#constant stores gpio pin used by dout pin of hx711. It will be used to check if hx711 is ready to send data or not


'''
*
* Function Name: 	calculateWeight
* Input: 			none
* Output: 			returns the calculated weight from the load cell value
* Logic: 			1) take the reading from load cell
*					2) take the difference between current value and base value
*					3) divide the difference with diference got with known weight
*					4) finally multiply the division answer with known weight value to get the weight
* Example Call:		calculateWeight()
*
'''
def caculateWeight():
	global taredWeight
	global baseValue
	val = lc.read_cell_value()  #read load cell value
	weight = ((baseValue - val) / 49000.0) * 230.0  #convert them into weight
	weight = weight - taredWeight  #remove tared weight from calculated weight
	if weight < 0:	#if weight becomes negative then set it back to zero
		weight = 0
	return int(weight)

# Show the weight from the load sensor to the screen.
def displayWeight():
	lcd.string("Object weight is:", LINE_3)
	weight = caculateWeight()	#get calculated weight from the calculateWeight function
	lcd.string(str(weight) + " grams", LINE_4)	#display the weight on the lcd
	if weight < min_weight:
		lcd.string("Place your object on", LINE_1)
		lcd.string("the platform", LINE_2)
	else:
		lcd.string("Press * button to", LINE_1)
		lcd.string("continue.", LINE_2)
	return weight


# Change the base value to current load sensor value so that weight is zero	
def tare():
	global baseValue
	global taredWeight
	lcd.clear()
	lcd.string("Taring weight...", LINE_1)
	lcval = lc.read_average_value(10)
	diff = math.fabs(baseValue- lcval)
	taredWeight = (diff / 49000.0) * 230.0  #store the calculated weight in variable


# Take a picture of the Crop from the webcam
def takePicture():
	lcd.string("Taking picture...", LINE_2)
	if os.path.exists('/dev/video0'):
		#create image file name with current date
		imgName = "image-"+ datetime.datetime.now().isoformat()+str(USER_ID)+".jpg"
		imagepath = "/home/pi/ghfarm/images/%s" %imgName
		#capture image and save in images directory. if image file does not exists in folder then retake the image
		while os.path.isfile(imagepath) == False:
			os.system("fswebcam -r 640x480 -S 10 --no-banner %s" %imagepath)
		return True, imgName
	else:	#if camera is not attached display error message
		lcd.clear()
		lcd.string("      FAILED", LINE_1)
		lcd.string("No camera attached", LINE_2)
		time.sleep(2)
		return False, "NoCamera"
		
# Store all data offline for later Upload
def storeData(data):
	lcd.clear()
	lcd.string("Storing data...", LINE_1)
	lcd.string("Weight: "+str(data['weight']), LINE_2)
	lcd.string("CropID: "+str(data['crop_id']), LINE_3)
	lcd.string("TroughID: "+str(data['troughid']), LINE_4)
	f = open('/home/pi/ghfarm/details.txt','a')
	t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	crops = {'weight':data['weight'],'crop_id':data['crop_id'],'time': t, 'imagename':data['imagename'], 'troughid':data['troughid']}
	crop_details = json.dumps(crops)
	f.write(crop_details +'\n')
	time.sleep(2)


def validateCropID():
	return True


# Accept a valid crop id as input from the keypad
def acceptCropID():
	lcd.clear()
	cropID = ""
	key = ""
	time.sleep(0.1)

	lcd.string(" Enter Crop ID  ", LINE_1)
	lcd.string("  *- continue   ", LINE_2)
	lcd.string("#- clear,D- back", LINE_3)
	#loop until some crop id is entered and * key is pressed. Following loop will run until valid crop id entered
	while True:
		while  key != "*":
			lcd.string(cropID, LINE_4)
			key = kpad.get_key()
			if key == '*':
				if len(cropID) == 0:
					lcd.clear()
					lcd.string("Crop ID cant", LINE_1)
					lcd.string("be null", LINE_2)
					time.sleep(1)
					lcd.clear()
					lcd.string(" Enter Crop ID  ", LINE_1)
					lcd.string("  *- continue   ", LINE_2)
					lcd.string("#- clear,D- back", LINE_3)
				else:
					break
			elif key == '#':  #for backspacing
				if len(cropID) > 0:
					cropID = cropID[:-1]
				time.sleep(0.1)
			elif key == 'D':  #return to previous menu
					return False, ""
			elif key.isdigit():
				cropID += key
				time.sleep(0.2)
			key = ""
		if validateCropID(): # verify trough id
				return True, cropID
		else:
			lcd.clear()
			lcd.string("Trough ID is", LINE_1)
			lcd.string("    Invalid!", LINE_2)
			lcd.string("Please Try Again", LINE_3)
			time.sleep(1)



def validateTroughID():
	return True



# Accept a valid trough id from user as input	
def acceptTroughID():
	lcd.clear()
	troughID = ""
	key = "E"
	time.sleep(0.1)
	lcd.string("Enter Trough ID", LINE_1)
	lcd.string("*- continue", LINE_2)
	lcd.string("#- clear D- back", LINE_3)
	#loop until some trough id is entered and * key is pressed. Following loop will only break when valid trough ID is entered
	while True:
		while  key != "*":
			lcd.string(troughID, LINE_4)
			key = kpad.get_key()
			if key == '*':
				if len(troughID) == 0:
					lcd.clear()
					lcd.string("Trough ID can't", LINE_1)
					lcd.string("be null", LINE_2)
					time.sleep(1)
					lcd.clear()
					lcd.string("Enter Trough ID", LINE_1)
					lcd.string("*- continue", LINE_2)
					lcd.string("#- clear D- back", LINE_3)
				else:
					break
			elif key == '#':  #for backspacing
				if len(troughID) > 0:
					troughID = troughID[:-1]
				time.sleep(0.1)
			elif key == 'D':  #return to previous menu
				return False, ""
			elif key.isdigit():
				troughID += key
				time.sleep(0.1)
			key = ""
		if validateTroughID(): # verify trough id
			return True, troughID
		else:
			lcd.clear()
			lcd.string("Trough ID is", LINE_1)
			lcd.string("    Invalid!", LINE_2)
			lcd.string("Please Try Again", LINE_3)
			time.sleep(1)


# The default display. Show the reading from load sensor on the screen
def display_screen():
	weight = 0
	while True:
		key="E"
		while key is not '*' :
			weight = displayWeight()
			key = kpad.get_key()
			if key == 'D' :
				tare()
			elif key == 'A':
				lcd.clear()
				lcd.string("       System", LINE_2)
				lcd.string("  Shutting down...", LINE_3)
				active = 0
				os.system("sudo poweroff")
				lcd.clear()
				break
			elif key == 'B':
				lcd.clear()
				lcd.string("       Script", LINE_2)
				lcd.string("     Restarting", LINE_3)
				lcd.string("   Please wait...", LINE_4)
				active = 0
				GPIO.cleanup()
				sys.stdout.flush()
				os.execv(sys.executable, ['python3'] + sys.argv)
				break
			elif key == 'C':
				lcd.clear()
				lcd.string("       System", LINE_2)
				lcd.string("     Restarting", LINE_3)
				lcd.string("   Please wait...", LINE_4)
				active = 0
				os.system("sudo reboot")
				break
		if(weight<min_weight):
			continue
		else:
			return weight


# Check if the machine is connected to the internet
def is_connected(url):
	try:
		urllib.request.urlopen(url, timeout=5)
		return True
	except:
		return False


# Send the crop image and other data to the server for preidction and return the predicted crop name and id
def predict(data):
	lcd.clear()
	lcd.string("       Asking", LINE_2)
	lcd.string("    Nostradamus", LINE_3)
	lcd.string("   Please wait...", LINE_4)
	data['user_id']=USER_ID
	data['password']=PASSWORD
	with open(imagepath + data['imagename'], 'rb') as img:
		image = img.read()
		image = str(image,"latin-1")
	data['image']=image
	try:
		r = requests.post(PREDICT_URL, data=json.dumps(data).encode('utf-8'))
		print(r.text)
		r = json.loads(r.text)
		return True, r[0], r[1], r[2]
	except Exception as e:
		return False, e, "", "", ""


# Send all the data along with the crop id to the server
def send_all_data(data):
	lcd.clear()
	lcd.string("Sending data...", LINE_1)
	lcd.string("Weight: "+str(data['weight']), LINE_2)
	lcd.string("CropID: "+str(data['crop_id']), LINE_3)
	lcd.string("TroughID: "+str(data['troughid']), LINE_4)
	time.sleep(1)
	data['user_id']=USER_ID
	data['password']=PASSWORD
	data['time']= datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	with open(imagepath + data['imagename'], 'rb') as img:
		image = img.read()
		image = str(image,"latin-1")
	data['image']=image
	try:
		r = requests.post(URL, data=json.dumps(data))
		print(r.text)
		if(r.text != "Done"):
			with open('/home/pi/ghfarm/details.txt', 'w') as detail_file:
				detail_file.write(json.dumps(data))
			return False
		return True
	except:
		with open('/home/pi/ghfarm/details.txt', 'w') as detail_file:
				detail_file.write(json.dumps(data))
		return False


# Generate 4 options at a time
def generate_options(names,percentages,primary_keys):
	all_options = []
	all_pks = []
	assert len(names) == len(percentages) , "Length's of names and percentages don't match!"
	for start_idx in range(0,len(names),4):
		s = []
		pk = []
		for j in range(0,min(4,len(names) - start_idx)):
			idx = start_idx + j
			assert len(names[idx]) <= 11, "Length of name too long!"
			row = str(j+1) + '.' + names[idx].ljust(11) + (' {:05.2f}'.rjust(5).format(percentages[idx]*100)) + '%'
			s.append(row)
			pk.append(primary_keys[idx])
		all_options.append(s)
		all_pks.append(pk)
	return all_options,all_pks

# Show choices to select crop along with percentages
def show_choices(names,percentages,primary_keys):
	lcd.clear()
	print("Options:")
	options,pks = generate_options(names,percentages,primary_keys)
	last_idx = len(options) - 1
	print(options)
	curr_start_idx = 0
	key = ''
	while True:
		option = options[curr_start_idx]
		for i in range(0,len(option)):
			lcd.string(option[i],LINES[i])

		for i in range(1,len(option)+1):
			key = kpad.get_key()
			if key == str(i):
				time.sleep(0.1)
				return pks[curr_start_idx][i-1]

		key = kpad.get_key()

		if key == '*':
			lcd.clear()
			time.sleep(0.1)
			if curr_start_idx == last_idx:
				curr_start_idx = 0
			elif curr_start_idx < last_idx:
				curr_start_idx = (curr_start_idx + 1)
			continue

		if key == '#':
			lcd.clear()
			time.sleep(0.1)
			if curr_start_idx == 0:
				curr_start_idx = last_idx
			elif curr_start_idx >= 1:
				curr_start_idx = (curr_start_idx - 1)
			continue
				

# Display the prediction on the screen and change is asked to.
def show_prediction(names, percentages,primary_keys):
	lcd.clear()
	time.sleep(0.3)
	s = ("Confidence: {:05.2f}".format(percentages[0]*100)) + '%'
	lcd.string(names[0], LINE_1)
	lcd.string(s, LINE_2)
	lcd.string("* to continue", LINE_3)
	lcd.string("# to change", LINE_4)
	key =""
	while True:
		if key == '*':
			return primary_keys[0]
		if key == '#':
			return show_choices(names,percentages,primary_keys)
		key = kpad.get_key()

def init():
	print("Initialization")
	global baseValue
	#initialize lcd
	lcd.lcd_init()
	lcd.string("      Welcome", LINE_1)
	lcd.string(" Remove any object", LINE_2)
	lcd.string(" from the platform.", LINE_3)
	time.sleep(2)
	lcd.clear()
	lcd.string("    Calibrating", LINE_1)
	lcd.string("   Please wait...", LINE_2)
	baseValue = lc.base_value()
try :
	init()
	print("Started System")
	lcd.string("Started System", LINE_1)
	data = {}	#Dictionary to store all required data
	stage = 0
	while True:
		data['weight'] = display_screen()
		print(data['weight'])
		while True:
			if stage==0:
				print("Taking Picture")
				imageAccepted,data['imagename'] = takePicture()
				if(imageAccepted):
					stage += 1
				else:
					break
			if stage==1:
				print("Taking TroughID")
				troughIDAccepted,data['troughid'] = acceptTroughID()
				if(troughIDAccepted):
					stage += 1 
				else:
					stage -= 1
					continue
			if stage==2:
				print("Trying to Connect")
				if is_connected(PREDICT_URL):
					print("Calling Predict")
					success, names, percentages, primary_keys = predict(data)
					data['crop_id'] = primary_keys[0]
					print(success,percentages,primary_keys)
					if success:
						print("Showing Prediction")
						data['crop_id'] = show_prediction(names,percentages,primary_keys)
					else:
						while True:
							cropIDAccepted, data['crop_id'] =  acceptCropID()
							if(cropIDAccepted):
								 break
							else:
								continue
				else:
					print("Taking Crop ID")
					while True:
						cropIDAccepted, data['crop_id'] =  acceptCropID()
						if(cropIDAccepted):
							 break
						else:
							continue
				stage += 1
			if stage==3:
				if is_connected(URL):
					print("Sending Data")
					send_all_data(data)
				else:
					print("Storing DAta")
					storeData(data)
				stage = 0
				break
except KeyboardInterrupt:
	print("\nInterrupted by keyboard")
except Exception as e:
	print("An Exception Occured of type {0}. Arguments:\n{1!r}".format(type(e).__name__,e.args))

finally:
	lcd.clear()
	time.sleep(1)
	GPIO.cleanup()
