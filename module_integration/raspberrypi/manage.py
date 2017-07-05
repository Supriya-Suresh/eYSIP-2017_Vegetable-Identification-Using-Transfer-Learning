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

#address constant for lines in lcd display
LINE_1 = 0x80
LINE_2 = 0xC0
LINE_3 = 0x94
LINE_4 = 0xD4

baseValue = 0	#variable to store the base value of load cell
taredWeight = 0	#variable to store tared weight
imgName = ""	#variable to store image name
measuredWeight = 0.0	#variable to store calculated weight
DOUT = 22	#constant stores gpio pin used by dout pin of hx711. It will be used to check if hx711 is ready to send data or not
troughID = ""	#variable to sotre trough ID
locationID = ""	#variable to sotre location ID
cropID = ""	#variable to store crop ID
cropName = ""	#variable to store crop name
locationName = ""	#variable to store location name

#Flag variables
troughIDExcepted = 0  #to check if trough id is accepted or not
locationIDExcepted = 0  #to check if location id is accepted or not
cropIDExcepted = 0	#to check if crop id is accepted or not
pictureTaken = 0	#to check if picture is taken or not
active = 1	#to check if program is active or not

#initialize lcd
lcd.lcd_init()

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
	global measuredWeight
	global baseValue
	val = lc.read_cell_value()  #read load cell value
	weight = ((baseValue - val) / 49000.0) * 230.0  #convert them into weight
	weight = weight - taredWeight  #remove tared weight from calculated weight
	if weight < 0:	#if weight becomes negative then set it back to zero
		weight = 0
	weight = int(weight)
	measuredWeight = weight	#store weight into measuredWeight variable
	return measuredWeight	#return the calculated weight

'''
*
* Function Name: 	displayWeight
* Input: 				none
* Output: 			none
* Logic: 				it displays weight on the lcd screen by using calculateWeight function
* Example Call:		displayWeight()
*
'''
def displayWeight() :
	global measuredWeight
	lcd.string("Object weight is:", LINE_3)
	weight = caculateWeight()	#get calculated weight from the calculateWeight function
	lcd.string(str(weight) + " grams", LINE_4)	#display the weight on the lcd
	if measuredWeight < 10:
		lcd.string("Place your object on", LINE_1)
		lcd.string("the platform", LINE_2)
	else:
		lcd.string("Press * button to", LINE_1)
		lcd.string("continue.", LINE_2)
	

'''
*
* Function Name: 	tare
* Input: 				none
* Output: 			none
* Logic: 				takes the current weight of the object and stores it in variable then it will be subtracted form current weight value
* Example Call:		tare()
*
'''
def tare():
	global baseValue
	global taredWeight
	lcd.clear()
	lcd.string("Taring weight...", LINE_1)
	lcval = lc.read_average_value(10)
	diff = math.fabs(baseValue- lcval)
	taredWeight = (diff / 49000.0) * 230.0  #store the calculated weight in variable

'''
*
* Function Name: 	takePicture
* Input: 				none
* Output: 			none
* Logic: 				takes picture using USB camera using fscam program
* Example Call:		takePicture()
*
'''
def takePicture():
	global imgName
	global pictureTaken
	lcd.string("Taking picture...", LINE_2)
	if os.path.exists('/dev/video0'):
		#create image file name with current date
		imgName = "image-" + datetime.datetime.now().isoformat() + ".jpg"
		imgName = "/home/pi/ghfarm/images/%s" %imgName
		#capture image and save in images directory. if image file does not exists in folder then retake the image
		while os.path.isfile(imgName) == False:
			os.system("fswebcam -r 640x480 -S 10 --no-banner /home/pi/ghfarm/images/%s" %imgName)
		pictureTaken = 1  #if picture is successfully taken then set pictureTaken flag to 1
	else:	#if camera is not attached display error message
		lcd.clear()
		lcd.string("      FAILED", LINE_1)
		lcd.string("No camera attached", LINE_2)
		time.sleep(2)
		######################################################################################################################
		pictureTaken = 1



'''
*
* Function Name: 	storeData
* Input: 				none
* Output: 			none
* Logic: 				stores the data into local database
* Example Call:		storeData()
*
'''
def storeData():
	global imgName
	lcd.string("Storing data...", LINE_3)
	f = open('/home/pi/ghfarm/details.txt','a')
	time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	crops = {'weight':measuredWeight,'crop_id':cropID,'time': time, 'imagepath':imgName, 'troughid':troughID}
	crop_details = json.dumps(crops)
	f.write(crop_details +'\n')

'''
*
* Function Name: 	fetchCropInfo
* Input: 				none
* Output: 			none
* Logic: 				fetches the crop name and id from local database
* Example Call:		fetchCropInfo()
*
'''
def fetchCropInfo():
	global cropName 
	global cropIDExcepted
	cropName = "Spinach"
	cropIDExcepted = 1
	# "retrieves crop info through cropid info"
	# global cropID
	# global cropName
	# global cropIDExcepted
	# try:
	# 	lcd.clear()
	# 	lcd.string("Fetching crop info...", LINE_1)
	# 	#create instance of a database with host, username, password and database name
	# 	db = sqldb.connect("localhost", "root", "firebird", "maindb")
	# 	#create cursor object
	# 	cursor = db.cursor()
	# 	cid = int(cropID)  #convert accepted crop id into integer
	# 	sql = "SELECT name FROM cropinfo WHERE id = %d" % (cid)
	# 	cursor.execute(sql)
	# 	data = cursor.fetchone()
	# 	#if there some crop exists with this id
	# 	if data > 0:
	# 		cropName = data[0]  #then assigh cropname to variable
	# 		cropIDExcepted = 1	#set cropIDExcepted flag to one
	# 		#lcd.clear()
	# 		#lcd.string("Successfully fetched", LINE_1)
	# 		#lcd.string("crop information", LINE_2)
	# 		#time.sleep(0.5)
	# 	else:  #if no crop exists with entered ID
	# 		lcd.clear()
	# 		lcd.string("Crop ID does not", LINE_1)
	# 		lcd.string("exists!", LINE_2)
	# 		lcd.string("Enter valid crop ID", LINE_3)
	# 		time.sleep(3)
	# except:  #if database connection Fails
	# 	lcd.clear()
	# 	lcd.string("      FAILED", LINE_1)
	# 	lcd.string("Unable to connect to", LINE_2)
	# 	lcd.string("local database", LINE_3)
	# 	lcd.string("Try again later", LINE_4)
	# 	time.sleep(3)
	# db.close()

'''
*
* Function Name: 	acceptCropID
* Input: 				none
* Output: 			none
* Logic: 				accepts crop ID from user using keypad
* Example Call:		acceptCropID()
*
'''
def acceptCropID():
	global cropID
	lcd.clear()
	cropID = ""
	key = ""
	time.sleep(0.1)

	lcd.string("Enter Crop ID", LINE_1)
	lcd.string("Press * to continue", LINE_2)
	lcd.string("and # for backspace", LINE_3)
	#loop until some crop id is entered and * key is pressed. Following loop will run until valid crop id entered
	while  key != "*":
		lcd.string(cropID, LINE_4)
		key = kpad.get_key()
		if key == '*':
			if len(cropID) <= 0:
				lcd.clear()
				lcd.string("Crop ID cant", LINE_1)
				lcd.string("be null", LINE_2)
				time.sleep(1)
				lcd.clear()
				lcd.string("Enter Crop ID", LINE_1)
				lcd.string("Press * to continue", LINE_2)
				lcd.string("and # for backspace", LINE_3)
			else:
				break
		elif key == '#':  #for backspacing
			if len(cropID) > 0:
				cropID = cropID[:-1]
		elif key.isdigit():
			cropID += key
			time.sleep(0.2)
		key = ""
	#after accepting crop ID fetch crop information from local database
	print("Calling fetchcrop")
	fetchCropInfo()


'''
*
* Function Name: 	fetchTroughInfo
* Input: 				none
* Output: 			none
* Logic: 				fetches the trough name and id from local database
* Example Call:		fetchTroughInfo()
*
'''
def fetchTroughInfo():
	global troughID
	global troughIDExcepted
	troughIDExcepted = 1
	
'''
*
* Function Name: 	acceptTroughID
* Input: 				none
* Output: 			none
* Logic: 				accepts trough ID from user using keypad
* Example Call:		acceptTroughID()
*
'''
def acceptTroughID():
	global troughID
	global troughIDExcepted
	lcd.clear()
	troughID = ""
	key = "E"
	time.sleep(0.1)
	lcd.string("Enter Trough ID", LINE_1)
	lcd.string("Press * to continue", LINE_2)
	lcd.string("and # for backspace", LINE_3)
	#loop until some trough id is entered and * key is pressed. Following loop will only break when valid trough ID is entered
	while  key != "*":
		lcd.string(troughID, LINE_4)
		key = kpad.get_key()
		if key == '*':
			if len(troughID) <= 0:
				lcd.clear()
				lcd.string("Trough ID can't", LINE_1)
				lcd.string("be null", LINE_2)
				time.sleep(1)
				lcd.clear()
				lcd.string("Enter Trough ID", LINE_1)
				lcd.string("Press * to continue", LINE_2)
				lcd.string("and # for backspace", LINE_3)
			else:
				break
		elif key == '#':  #for backspacing
			if len(troughID) > 0:
				troughID = troughID[:-1]
		elif key.isdigit():
			troughID += key
			time.sleep(0.1)
		key = ""
	#check if entered trough ID is valid or not by fetching it from local database
	fetchTroughInfo()


'''
*
* Function Name: 	init
* Input: 				none
* Output: 			none
* Logic: 				calculates the baseValue of load cell and fetches the crop info from the server database
* Example Call:		init()
*
'''
def init():
	print("Initialization")
	global baseValue
	lcd.string("      Welcome", LINE_1)
	lcd.string(" Remove any object", LINE_2)
	lcd.string(" from the platform.", LINE_3)
	time.sleep(2)
	lcd.clear()
	lcd.string("      Welcome", LINE_1)
	lcd.string("   Please wait...", LINE_2)
	baseValue = lc.base_value()

try :
	init()
	print("Started System")
	lcd.string("Started System", LINE_1)
	troughIDExcepted = cropIDExcepted = pictureTaken = 0
	key = "E"

	#while key pressed is not the * key
	while True:
		while key is not '*' :
			displayWeight()
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
				os.execv(sys.executable, ['python'] + sys.argv)
				break
			elif key == 'C':
				lcd.clear()
				lcd.string("       System", LINE_2)
				lcd.string("     Restarting", LINE_3)
				lcd.string("   Please wait...", LINE_4)
				active = 0
				os.system("sudo reboot")
				break

		acceptCropID()
		print("Accepted Crop")
		if cropIDExcepted:
			print("Calling acceptTrough")
			acceptTroughID()
		if troughIDExcepted:
			print("Calling acceptLocation")
			takePicture()
		if pictureTaken:
			print("Calling Store Data")
			storeData()
		key = "E"
except KeyboardInterrupt:
	print("Interrupted by keyboard")
finally:
	lcd.clear()
	time.sleep(1)
	GPIO.cleanup()
