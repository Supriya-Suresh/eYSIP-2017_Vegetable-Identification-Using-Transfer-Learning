import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

KEYPAD = [ ['1','2','3','A'],
['4','5','6','B'],
['7','8','9','C'],
['*','0','#','D']
]

ROW=[29,31,33,35]
COLUMN=[18,40, 38, 36]

MAX = 50
FRAC = 3

def get_key ():

	row = [0,0,0,0]
	col = [0,0,0,0]

	for z in range(MAX):
		for j in range(len(COLUMN)):
			GPIO.setup(COLUMN[j], GPIO.OUT)
			GPIO.output(COLUMN[j], GPIO.LOW)
		 
		# Set all rows as input
		for i in range(len(ROW)):
			GPIO.setup(ROW[i], GPIO.IN, pull_up_down=GPIO.PUD_UP)
		 
		# Scan rows for pushed key/button
		# A valid key press should set "rowVal"  between 0 and 3.
		rowVal = -1
		for i in range(len(ROW)):
			tmpRead = GPIO.input(ROW[i])
			if tmpRead == 0:
				row[i] += 1

		# Convert columns to input
		for j in range(len(COLUMN)):
			GPIO.setup(COLUMN[j], GPIO.IN)
		 
		# Switch the i-th row found from scan to output
		GPIO.setup(ROW[rowVal], GPIO.OUT)
		GPIO.output(ROW[rowVal], GPIO.HIGH)

		# Scan columns for still-pushed key/button
		# A valid key press should set "colVal"  between 0 and 3.
		colVal = -1
		for j in range(len(COLUMN)):
			tmpRead = GPIO.input(COLUMN[j])
			if tmpRead == 1:
				col[j] += 1

		rowMax = max(row)
		colMax = max(col)
		rowVal = row.index(rowMax)
		colVal = col.index(colMax)

	exit_cust()
	if rowMax>(MAX/FRAC) and colMax>(MAX/FRAC):
		print("KEY: "+KEYPAD[rowVal][colVal])
		return KEYPAD[rowVal][colVal]
	return "E"

def exit_cust():
    # Reinitialize all rows and columns as input at exit
    for i in range(len(ROW)):
            GPIO.setup(ROW[i], GPIO.IN, pull_up_down=GPIO.PUD_UP) 
    for j in range(len(COLUMN)):
            GPIO.setup(COLUMN[j], GPIO.IN, pull_up_down=GPIO.PUD_UP)
    time.sleep(0.1)